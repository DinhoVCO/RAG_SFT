from prompts.teleqna_prompts import get_full_promt_teleqna
from prompts.clapnq_prompts import get_full_promt_clapnq
from prompts.boolq_prompts import get_full_promt_boolq
from prompts.covid_prompts import get_full_promt_covid
from utils.datasets_splits import load_dataset_splits

def get_dataset_for_train_phi(dataset_name, include_docs=False, vector_store=None, top_k=4, batch_size=8):
    train_dataset, val_dataset, test_dataset = load_dataset_splits(dataset_name)
    if(dataset_name=="teleqna"):
        print(f"Creating dataset for {dataset_name}")
        return get_dataset_for_training_teleqna(train_dataset, include_docs, vector_store, top_k, batch_size)
    elif(dataset_name=="squad"):
        return 0
    elif(dataset_name=="clapnq"):
        return get_dataset_for_training_clapnq(train_dataset, include_docs, vector_store, top_k, batch_size)
    elif(dataset_name=="boolq"):
        return get_dataset_for_training_boolq(train_dataset, include_docs, vector_store, top_k, batch_size)
    elif(dataset_name=="covid"):
        return get_dataset_for_training_covid(train_dataset, include_docs, vector_store, top_k, batch_size)
    else:
        raise ValueError(f"Incorrect dataset name: {name_dataset}")

def add_relevant_docs(dataset, vector_store, top_k, batch_size):
    retrieval_docs= vector_store.buscar_por_batches(dataset['question'], top_k=top_k, batch_size=batch_size)
    relevant_documents=[]
    for docs in retrieval_docs:
        relevant_documents.append(docs[0])
    dataset = dataset.add_column("relevant_documents", relevant_documents)
    return dataset

def final_dataset_for_training(dataset, get_full_promt, include_docs):
    new_dataset = dataset.map(
        lambda row: {'text': get_full_promt(row, include_docs)},
        remove_columns=dataset.column_names
    )
    return new_dataset

def get_dataset_for_training_teleqna(dataset, include_docs, vector_store, top_k, batch_size):
    if(include_docs):
        dataset= add_relevant_docs(dataset, vector_store, top_k, batch_size) 
    final_dataset = final_dataset_for_training(dataset, get_full_promt_teleqna, include_docs)
    return final_dataset
    
def get_dataset_for_training_clapnq(dataset, include_docs, vector_store, top_k, batch_size):
    dataset = dataset.rename_column("input", "question")
    if(include_docs):
        dataset= add_relevant_docs(dataset, vector_store, top_k, batch_size) 
    final_dataset = final_dataset_for_training(dataset, get_full_promt_clapnq, include_docs)
    return final_dataset

def get_dataset_for_training_boolq(dataset, include_docs, vector_store, top_k, batch_size):
    if(include_docs):
        dataset= add_relevant_docs(dataset, vector_store, top_k, batch_size) 
    final_dataset = final_dataset_for_training(dataset, get_full_promt_boolq, include_docs)
    return final_dataset

def get_dataset_for_training_covid(dataset, include_docs, vector_store, top_k, batch_size):
    if(include_docs):
        dataset= add_relevant_docs(dataset, vector_store, top_k, batch_size) 
    final_dataset = final_dataset_for_training(dataset, get_full_promt_covid, include_docs)
    return final_dataset