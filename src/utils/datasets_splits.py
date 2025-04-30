from datasets import load_dataset

def load_dataset_splits(name_dataset):
    print(f"Loading dataset splits for {name_dataset}")
    if(name_dataset=="teleqna"):
        return dataset_for_teleqna()
    elif(name_dataset=="squad"):
        return get_passages_for_squad()
    elif(name_dataset=="clapnq"):
        return get_passages_for_clapnq()
    elif(name_dataset=="boolq"):
        return get_passages_for_boolq()
    elif(name_dataset=="covid"):
        return get_passages_for_covid()
    else:
        raise ValueError(f"Incorrect dataset name: {name_dataset}")

def dataset_for_teleqna():
    DATASET_NAME = 'DinoStackAI/3GPP-QA-MultipleChoice'
    dataset = load_dataset(DATASET_NAME)
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_squad():
    DATASET_NAME = 'rajpurkar/squad'
    dataset_all = load_dataset(DATASET_NAME, split='train')
    dataset_dict = dataset_all.train_test_split(test_size=0.80, seed=42)
    dataset = dataset_dict['train']
    new_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    final_dataset = new_dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = final_dataset['train']
    val_dataset = final_dataset['test']
    test_dataset = new_dataset['test']
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_clapnq():    
    DATASET_NAME = 'PrimeQA/clapnq'
    dataset = load_dataset(DATASET_NAME)
    temp_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = temp_dataset['train']
    val_dataset = temp_dataset['test']
    test_dataset = dataset['validation']
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_boolq():
    print("Loading and preparing datasets...")
    DATASET_NAME = 'google/boolq'
    dataset= load_dataset(DATASET_NAME)
    temp_dataset = dataset['train'].train_test_split(test_size=0.20, seed=42)
    train_dataset = temp_dataset['train']
    val_dataset = temp_dataset['test']
    test_dataset = dataset['validation']
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_covid():
    DATASET_NAME = 'deepset/covid_qa_deepset'
    dataset= load_dataset(DATASET_NAME,split='train')
    temp_dataset = dataset.train_test_split(test_size=0.20, seed=42)
    dev_dataset = temp_dataset['train'].train_test_split(test_size=0.20, seed=42)
    train_dataset = dev_dataset['train']
    val_dataset = dev_dataset['test']
    test_dataset = temp_dataset['test']
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset