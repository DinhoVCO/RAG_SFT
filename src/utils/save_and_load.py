from langchain_core.documents import Document
import json
from typing import Iterable, Any

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

def save_config( config: dict, path: str) -> None:
    try:
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
    except (IOError, TypeError) as e:
        print(f"Error saving config to {path}: {e}")

def load_config(path: str) -> dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading config from {path}: {e}")
        return {}