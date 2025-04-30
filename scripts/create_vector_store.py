import sys
import os
sys.path.append('../src')
from utils.get_documents import get_passages_by_dataset
from vector_stores.faiss import VectorStoreFaiss
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
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

def create_vector_store(passages, embedding_model, folder_name, path_to_save, batch_size=2048):
    vector_store = VectorStoreFaiss.from_documents(embedding_model, passages, batch_size)
    vector_store.save_local(f'{path_to_save}{folder_name}')
    return vector_store


def main():
    args = parse_arguments()
    folder_name = f"vs_{args.dataset}_{args.cs}_{args.co}"
    dir_path = os.path.join(args.output_dir, folder_name)
    if os.path.isdir(dir_path):
        print("The vector store already exists")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Load model embedding")
        print(f"Using device: {device}")
        emb_tok = AutoTokenizer.from_pretrained(args.emb_model)
        passages = get_passages_by_dataset(args.dataset, args.cs, args.co, emb_tok)
        print("Creating vector store")
        create_vector_store(passages, args.emb_model, folder_name, args.output_dir, args.bs_emb)
    
if __name__ == "__main__":
    main()

