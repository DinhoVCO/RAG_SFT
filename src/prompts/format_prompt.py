from prompts.teleqna_prompts import get_full_promt_teleqna


def get_prompt_for_train_phi(dataset_name, row, include_docs):
    if(dataset_name=="teleqna"):
        return get_full_promt_teleqna(row, include_docs)
    elif(dataset_name=="teleqna"):
        
    elif(dataset_name=="teleqna"):
    elif(dataset_name=="teleqna"):
    elif(dataset_name=="teleqna"):
    else: 
        raise ValueError(f"Incorrect dataset name: {name_dataset}")