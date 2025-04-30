import sys
sys.path.append('../src')
from vector_stores.faiss import VectorStoreFaiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from rag.rag_basic import RAGPipeline
import pandas as pd
import argparse
from utils.datasets_splits import load_dataset_splits
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_name, device):
    print("loading model base")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def load_reader_ft_model(model_name, device, lora_adapter_path=None):
    model, tokenizer = load_model(model_name, device)
    if(lora_adapter_path):
        print("loading adapter")
        model = PeftModel.from_pretrained(model, lora_adapter_path).to(device)
    return model, tokenizer


def create_rag_pipeline(dataset_name, reader_model, tokenizer, vector_store, max_new_tokens=None):
    reader_llm = pipeline(
        model=reader_model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=max_new_tokens,
    )
    return RAGPipeline(dataset_name, llm=reader_llm, knowledge_index=vector_store)

def load_dataset_for_inference(dataset_name):
    train_ds, val_ds, test_ds = load_dataset_splits(dataset_name)
    return test_ds

def save_answers_to_csv(inference, file_path):
    df = pd.DataFrame({
        'inference': inference,
    })
    df.to_csv(file_path, index=False)

def column_question(dataset_name):
    if dataset_name=='clapnq':
        return 'input'
    else:
        return 'question'

def main(emb_model, gen_model, lora_adapter_path, max_new_tokens, use_rag, vector_store_path, dataset_name, output_csv_path, bs_emb, bs_gen, top_k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    reader_model, tokenizer = load_reader_ft_model(gen_model, device, lora_adapter_path)
    vector_store = None
    if(use_rag):
        embedding_model = SentenceTransformer(emb_model, device=device)
        vector_store = VectorStoreFaiss.load_local(embedding_model, vector_store_path)
    rag_pipeline = create_rag_pipeline(dataset_name,reader_model, tokenizer, vector_store, max_new_tokens)
    dataset = load_dataset_for_inference(dataset_name)
    answers = rag_pipeline.answer_batch(dataset,column_question(dataset_name), bs_emb, bs_gen, top_k)
    save_answers_to_csv(answers, output_csv_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Execution")
    parser.add_argument("--emb_model", type=str, default='BAAI/bge-small-en-v1.5', help="Name of the embedding model")
    parser.add_argument("--gen_model", type=str, default='microsoft/phi-2', help="Name of the generative model")
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Name or path of the peft model")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens")
    parser.add_argument('--use_rag', action='store_true', help='Use RAG?')
    parser.add_argument("--vector_store_path", type=str, required=False, help="Path to the FAISS vector store")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of dataset (covid, clapnq, boolq, teleqna)")
    parser.add_argument("--output_csv_path", type=str, required=True, help="Path to save the answer output CSV")
    parser.add_argument("--bs_emb", type=int, default=100, help="batch size for encode dataset")
    parser.add_argument("--bs_gen", type=int, default=20, help="batch size for the LLM model")
    parser.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve from the index")

    args = parser.parse_args()

    main(
        args.emb_model,
        args.gen_model,
        args.lora_adapter_path,
        args.max_new_tokens,
        args.use_rag,
        args.vector_store_path,
        args.dataset_name,
        args.output_csv_path,
        args.bs_emb,
        args.bs_gen,
        args.top_k,
    )