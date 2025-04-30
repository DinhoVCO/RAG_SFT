from utils.save_and_load import load_docs_from_jsonl
from datasets import concatenate_datasets, load_dataset
from langchain_core.documents import Document
from utils.pre_processing_docs import remove_duplicates, create_text_splitter, split_documents
from tqdm import tqdm

def get_passages_by_dataset(name_dataset, chunk_size=150, chunk_overlap=20, tokenizer=None):
    if(name_dataset=="teleqna"):
        return get_passages_for_teleqna(chunk_size, chunk_overlap, tokenizer)
    elif(name_dataset=="squad"):
        return get_passages_for_squad()
    elif(name_dataset=="clapnq"):
        return get_passages_for_clapnq(chunk_size, chunk_overlap, tokenizer)
    elif(name_dataset=="boolq"):
        return get_passages_for_boolq(chunk_size, chunk_overlap, tokenizer)
    elif(name_dataset=="covid"):
        return get_passages_for_covid(chunk_size, chunk_overlap, tokenizer)
    else:
        raise ValueError(f"Incorrect dataset name: {name_dataset}")


def get_passages_for_teleqna(chunk_size, chunk_overlap, tokenizer):
    docs_dir = "../datasets/teleqna/corpus/3gpp_rel18_documents.jsonl"
    documents = load_docs_from_jsonl(docs_dir)
    print(f"Total number of documents: {len(documents)}")
    text_splitter = create_text_splitter(chunk_size, chunk_overlap, tokenizer)
    print(f"Creating passages")
    passages = split_documents(documents, text_splitter)
    print(f"Number of documents: {len(passages)}")
    print(f"Removing duplicate passages")
    passages_unique = remove_duplicates(passages)
    print(f"Total number of passages created: {len(passages_unique)}")
    return passages_unique

def get_passages_for_squad():
    def row_to_document(row: dict) -> Document:
        return Document(
            page_content=row["context"],
            metadata={"title": row["title"]}
        )
    DATASET_NAME="rajpurkar/squad"
    dataset = load_dataset(DATASET_NAME)
    dataset_combinado = concatenate_datasets([dataset['train'], dataset['validation']])
    passages = []
    for doc in dataset_combinado:
        passage = row_to_document(doc)
        passages.append(passage)
    print(f"Total number of passages: {len(passages)}")
    print(f"Removing duplicate passages")
    passages_unique = remove_duplicates(passages)
    print(f"Total number of passages created: {len(passages_unique)}")
    return passages_unique

def get_passages_for_covid(chunk_size, chunk_overlap, tokenizer):
    def row_to_document(row: dict) -> Document:
        return Document(
            page_content=row["context"],
            metadata={"doc_id": row["document_id"]}
        )
    DATASET_NAME="deepset/covid_qa_deepset"
    dataset = load_dataset(DATASET_NAME, split='train')
    documents = []
    for doc in dataset:
        document = row_to_document(doc)
        documents.append(document)
    print(f"Total number of documents: {len(documents)}")
    text_splitter = create_text_splitter(chunk_size, chunk_overlap, tokenizer)
    print(f"Creating passages")
    passages = split_documents(documents, text_splitter)
    print(f"Total number of passages: {len(passages)}")
    print(f"Removing duplicate passages")
    passages_unique = remove_duplicates(passages)
    print(f"Total number of passages created: {len(passages_unique)}")
    return passages_unique

# def get_passages_for_boolq(chunk_size, chunk_overlap, tokenizer):
#     def row_to_document(row: dict) -> Document:
#         return Document(
#             page_content=row["passage"]
#         )
#     DATASET_NAME="google/boolq"
#     dataset = load_dataset(DATASET_NAME)
#     dataset_combinado = concatenate_datasets([dataset['train'], dataset['validation']])
#     text_splitter = create_text_splitter(chunk_size, chunk_overlap, tokenizer)
#     passages = []
#     for row in dataset_combinado:
#         text_length = len(row['text'])
#         if(text_length>=150):
#             doc = row_to_document(row)
#             doc_passages = text_splitter.split_documents([doc])
#             passages += doc_passages
#         else:
#             passage = row_to_document(row)
#             passages.append(passage)
#     print(f"Total number of passages: {len(passages)}")
#     print(f"Removing duplicate passages")
#     passages_unique = remove_duplicates(passages)
#     print(f"Total number of passages created: {len(passages_unique)}")
#     return passages_unique


def get_passages_for_boolq(chunk_size, chunk_overlap, tokenizer):
    def row_to_document(row: dict) -> Document:
        return Document(
            page_content=row["passage"]
        )
    DATASET_NAME="google/boolq"
    dataset = load_dataset(DATASET_NAME)
    dataset_combinado = concatenate_datasets([dataset['train'], dataset['validation']])
    text_splitter = create_text_splitter(chunk_size, chunk_overlap, tokenizer)
    documents = []
    for row in tqdm(dataset_combinado):
        doc = row_to_document(row)
        documents.append(doc)
    passages = split_documents(documents, text_splitter)
    print(f"Total number of passages: {len(passages)}")
    print(f"Removing duplicate passages")
    passages_unique = remove_duplicates(passages)
    print(f"Total number of passages created: {len(passages_unique)}")
    return passages_unique



def get_passages_for_clapnq(chunk_size, chunk_overlap, tokenizer):
    def row_to_document(row: dict) -> Document:
        return Document(
            page_content=f"{row['title']}, {row['text']}",
            metadata={"title": row["title"], "doc_id":row["id"]}
        )
    DATASET_NAME="PrimeQA/clapnq_passages"
    dataset = load_dataset(DATASET_NAME, split='train')
    text_splitter = create_text_splitter(chunk_size, chunk_overlap, tokenizer)
    documents = []
    for row in tqdm(dataset):
        doc = row_to_document(row)
        documents.append(doc)
    passages = split_documents(documents, text_splitter)
    print(f"Total number of passages: {len(passages)}")
    print(f"Removing duplicate passages")
    passages_unique = remove_duplicates(passages)
    print(f"Total number of passages created: {len(passages_unique)}")
    return passages_unique


    

# def get_passages_for_clapnq(chunk_size, chunk_overlap, tokenizer):
#     def row_to_document(row: dict) -> Document:
#         return Document(
#             page_content=f"{row['title']}, {row['text']}",
#             metadata={"title": row["title"], "doc_id":row["id"]}
#         )
#     DATASET_NAME="PrimeQA/clapnq_passages"
#     dataset = load_dataset(DATASET_NAME, split='train')
#     text_splitter = create_text_splitter(chunk_size, chunk_overlap, tokenizer)
#     passages = []
#     for row in tqdm(dataset):
#         text_length = len(row['text'])
#         if(text_length>=150):
#             doc = row_to_document(row)
#             doc_passages = text_splitter.split_documents([doc])
#             passages += doc_passages
#         else:
#             passage = row_to_document(row)
#             passages.append(passage)
#     print(f"Total number of passages: {len(passages)}")
#     print(f"Removing duplicate passages")
#     passages_unique = remove_duplicates(passages)
#     print(f"Total number of passages created: {len(passages_unique)}")
#     return passages_unique




 