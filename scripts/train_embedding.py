import os
import sys
sys.path.append('../src')
import argparse
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from utils.data_for_train_emb import load_and_prepare_datasets
from utils.ir_evaluator import create_evaluator_information_retrieval
from dotenv import load_dotenv
import wandb


def parse_arguments():
    parser = argparse.ArgumentParser(description="Embedding model training")
    parser.add_argument('--name_dataset', type=str, required=True, help="Dataset name(clapnq, teleqna ")
    parser.add_argument('--model_name', type=str, required=True, help="Base model name")
    parser.add_argument('--new_model_name', type=str, required=True, help="Base new model name")
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--output_dir', type=str, required=True , help="Output directory for the trained model")
    args = parser.parse_args()
    return args


def initialize_model(model_name):
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    loss = MultipleNegativesRankingLoss(model)
    print("Model and loss function initialized.")
    return model, loss

def configure_training(my_model_name, epochs, batch_size, output_dir):
    print("Configuring training arguments...")
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir+f"/{my_model_name}",
        # Optional training parameters:
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        learning_rate= 5e-5,#2e-05, #3e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine_with_restarts",
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=15,
        save_strategy="steps",
        save_steps=15,
        save_total_limit=1,
        logging_steps=15,
        run_name=my_model_name,  # Will be used in W&B if wandb is installed
        load_best_model_at_end=True,
        metric_for_best_model="eval_telecom-ir-eval_cosine_mrr@10",
        report_to= "wandb" #"none" ,
    )
    print("Training arguments configured.")
    return args


def train_model(model, args_training, train_dataset, val_dataset, loss, evaluator):
    print("Starting training process...")
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args_training,
        train_dataset=train_dataset.select_columns(["question", "relevant_docs"]),
        eval_dataset=val_dataset.select_columns(["question", "relevant_docs"]),
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    print("Training completed.")
    return trainer

def save_model(trainer, output_dir, model_name):
    print("Saving the trained model...")
    trainer.model.save_pretrained(os.path.join(output_dir, model_name))
    print("Model saved successfully.")

# Main function
def main():
    args = parse_arguments()
    load_dotenv()
    wandb.login() 
    print("Starting main process...")
    train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(args.name_dataset)
    evaluator = create_evaluator_information_retrieval(args.name_dataset, val_dataset)
    my_model_name = f"{str(args.new_model_name)}_{str(args.epochs)}e_{str(args.batch_size)}bs"
    wandb.init(
        project="SBBD_embeddings",
        name=my_model_name,
    )
    model, loss = initialize_model(args.model_name)
    args_training = configure_training(my_model_name, args.epochs, args.batch_size, args.output_dir)
    trainer = train_model(model, args_training, train_dataset, val_dataset, loss, evaluator)
    save_model(trainer, args.output_dir, my_model_name)
    print("Process completed successfully.")

if __name__ == "__main__":
    main()