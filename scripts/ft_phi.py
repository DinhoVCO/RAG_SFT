import os
import sys
sys.path.append('../src')
import argparse
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from utils.data_for_train_phi import get_dataset_for_train_phi
from dotenv import load_dotenv
import wandb
from vector_stores.faiss import VectorStoreFaiss
from sentence_transformers import SentenceTransformer
import torch
from peft import get_peft_model

#5e 10bs

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-2 model with LoRA.")
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--new_model_name', type=str, default='phi-2-3GPP-RAG-ft' , help='Name for the fine-tuned model')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name for the dataset for training( covid, teleqna, coolq')
    parser.add_argument('--include_docs', action='store_true', help='Use RAG?')
    parser.add_argument('--top_k', type=int, required=True, help='Use retrieval top-k documents')
    parser.add_argument("--vector_store_path", type=str, default=None, help="Path to the FAISS vector store")
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the fine-tuned model')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument("--emb_model", type=str, default="", help="Name or Path of the embedding model")
    args = parser.parse_args()
    return args

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def configure_lora():
    return LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["Wqkv", "fc1", "fc2"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

def configure_training_arguments(output_dir, batch_size, num_epochs):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        fp16=True,
        bf16=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        learning_rate=2e-4,
        weight_decay=0.001,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        save_steps=100,
        save_total_limit=2,
        logging_steps=25,
        report_to="wandb",
    )

def train_model(batch_size, model_name, new_model_name, save_path, num_epochs,train_dataset_name,include_docs,top_k, vector_store_path, embedding_model):
    if(include_docs):
        vector_store = VectorStoreFaiss.load_local(embedding_model, vector_store_path)
    else:
        vector_store=None
    print("Generating dataset")
    print(f"Using k = {top_k} passages")
    dataset = get_dataset_for_train_phi(train_dataset_name, include_docs, vector_store,top_k, 8)
    print("Example")
    print(dataset[0]['text'])
    print("Loading model")
    model, tokenizer = load_model_and_tokenizer(model_name)
    peft_config = configure_lora()
    collator = DataCollatorForCompletionOnlyLM("Output:", tokenizer=tokenizer)
    output_dir = os.path.join(save_path, new_model_name)
    training_arguments = configure_training_arguments(
       output_dir, batch_size, num_epochs
    )
    print("Training")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_arguments,
        peft_config=peft_config,
        data_collator=collator,
    )
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")] if os.path.exists(output_dir) else []
    if checkpoints:
        print("🔵 Checkpoints detected. Resuming training from the last one.")
        print(checkpoints)
        trainer.train(resume_from_checkpoint=True)
    else:
        print("🟢 No checkpoints found. Starting training from scratch.")
        trainer.train()
    trainer.model.save_pretrained(os.path.join(save_path, "final"+new_model_name))
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Parámetros entrenables finales: {trainable_params}")

def main():
    args = parse_args()
    load_dotenv()
    wandb.login() 
    wandb.init(
        project="SBBD_phi-2-adapters",
        name=args.new_model_name,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    embedding_model = SentenceTransformer(args.emb_model, device=device)
    train_model(
        batch_size=args.batch_size,
        model_name='microsoft/phi-2',
        new_model_name=args.new_model_name,
        save_path=args.save_path,
        num_epochs=args.num_epochs,
        train_dataset_name=args.dataset_name,
        include_docs = args.include_docs,
        top_k =args.top_k,
        vector_store_path = args.vector_store_path,
        embedding_model = embedding_model
    )

if __name__ == "__main__":
    main()