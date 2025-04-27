import sys
import os
sys.path.append('../src')
from utils.save_and_load import save_docs_to_jsonl, load_docs_from_jsonl
from utils.pre_processing_docs import split_documents, remove_duplicates, create_text_splitter
from utils.get_documents import get_passages_by_dataset
from vector_stores.faiss import VectorStoreFaiss

from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Create vector store")
    parser.add_argument('--dataset', type=str, required=True, help="Name dataset (teleqna, squad)")
    parser.add_argument('--emb_model', type=str, required=True, help="Model name or path for embeddings")
    parser.add_argument('--cs', type=int, default=150, help="Chunk size")
    parser.add_argument('--co', type=int, default=20, help="Chunk overlap")
    parser.add_argument('--bs_emb', type=int, default=2048, help="Batch size for embedding")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save vector store")
    return parser.parse_args()

def create_vector_store(passages, chunk_size, chunk_overlap, embedding_model, tokenizer, folder_name, path_to_save, batch_size=2048):
    vector_store = VectorStoreFaiss.from_documents(embedding_model, passages, batch_size)
    vector_store.save_local(f'{path_to_save}{folder_name}')
    return vector_store


def main():
    args = parse_arguments()
    folder_name = f"vector_store_{args.dataset}"
    dir_path = args.output_dir+folder_name
    if os.path.isdir(dir_path):
        print("The vector store already exists")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Load model embedding")
        print(f"Using device: {device}")
        emb_tok = AutoTokenizer.from_pretrained(args.emb_model)
        passages = get_passages_by_dataset(args.dataset, args.cs, args.co, emb_tok)
        emb_model = SentenceTransformer(args.emb_model, device=device)
        print("Creating vector store")
        vector_store = create_vector_store(passages, args.cs, args.co, emb_model, emb_tok, folder_name, args.output_dir, args.bs_emb)
    
if __name__ == "__main__":
    main()

