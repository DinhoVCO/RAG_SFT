from langchain_core.documents import Document
from langchain.text_splitter import TextSplitter
from tqdm import tqdm
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(raw_documents:List[Document], text_splitter:TextSplitter) -> List[Document] :
    processed_docs = []
    for doc in tqdm(raw_documents):
        processed_docs += text_splitter.split_documents([doc])
    return processed_docs

def remove_duplicates(docs:List[Document])-> List[Document]:
    unique_texts = {}
    unique_docs = []
    for doc in docs:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            unique_docs.append(doc)
    return unique_docs

def create_text_splitter(chunk_size, chunk_overlap, tokenizer):
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", ",", ";", " ", ""]
    )