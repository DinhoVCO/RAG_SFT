from datasets import load_dataset

def load_and_prepare_datasets(name_dataset):
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
    print("Loading and preparing datasets...")
    DATASET_NAME = 'DinoStackAI/3GPP-QA-MultipleChoice'
    dataset = load_dataset(DATASET_NAME)
    train_dataset = dataset['train']
    train_dataset = train_dataset.select_columns(['question_id', 'question', 'explanation'])
    train_dataset = train_dataset.rename_columns({
        "question_id": "q_id",
        "question": "question",
        "explanation": "relevant_docs"
    })
    val_dataset = dataset['val']
    val_dataset = val_dataset.select_columns(['question_id', 'question', 'explanation'])
    val_dataset = val_dataset.rename_columns({
        "question_id": "q_id",
        "question": "question",
        "explanation": "relevant_docs"
    })
    test_dataset = dataset['test']
    test_dataset = test_dataset.select_columns(['question_id', 'question', 'explanation'])
    test_dataset = test_dataset.rename_columns({
        "question_id": "q_id", 
        "question": "question", 
        "explanation": "relevant_docs"
    })
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_squad():
    print("Loading and preparing datasets...")
    DATASET_NAME = 'rajpurkar/squad'
    dataset_all = load_dataset(DATASET_NAME, split='train')
    dataset_dict = dataset_all.train_test_split(test_size=0.80, seed=42)
    dataset = dataset_dict['train']
    new_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    final_dataset = new_dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = final_dataset['train']
    train_dataset = train_dataset.select_columns(['id', 'question', 'context'])
    train_dataset = train_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "context": "relevant_docs"
    })
    val_dataset = final_dataset['test']
    val_dataset = val_dataset.select_columns(['id', 'question', 'context'])
    val_dataset = val_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "context": "relevant_docs"
    })
    test_dataset = new_dataset['test']
    test_dataset = test_dataset.select_columns(['id', 'question', 'context'])
    test_dataset = test_dataset.rename_columns({
        "id": "q_id", 
        "question": "question", 
        "context": "relevant_docs"
    })
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_clapnq():
    def extract_title_and_text(example):
        if isinstance(example['passages'], list) and len(example['passages']) > 0:
            entry = example['passages'][0]
            title = entry.get('title', '')
            text = entry.get('text', '')
            return {'context': f"{title}, {text}"}
        return {'context': ''}
    
    print("Loading and preparing datasets...")
    DATASET_NAME = 'PrimeQA/clapnq'
    dataset = load_dataset(DATASET_NAME)
    temp_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
    train_dataset = temp_dataset['train']
    train_dataset = train_dataset.map(extract_title_and_text)
    train_dataset = train_dataset.select_columns(['id', 'input', 'context'])
    train_dataset = train_dataset.rename_columns({
        "id": "q_id",
        "input": "question",
        "context": "relevant_docs"
    })
    val_dataset = temp_dataset['test']
    val_dataset = val_dataset.map(extract_title_and_text)
    val_dataset = val_dataset.select_columns(['id', 'input', 'context'])
    val_dataset = val_dataset.rename_columns({
        "id": "q_id",
        "input": "question",
        "context": "relevant_docs"
    })
    test_dataset = dataset['validation']
    test_dataset = test_dataset.map(extract_title_and_text)
    test_dataset = test_dataset.select_columns(['id', 'input', 'context'])
    test_dataset = test_dataset.rename_columns({
        "id": "q_id", 
        "input": "question", 
        "context": "relevant_docs"
    })
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_boolq():
    def add_id(example, idx):
        return {"id": idx}
    print("Loading and preparing datasets...")
    DATASET_NAME = 'google/boolq'
    dataset= load_dataset(DATASET_NAME)
    temp_dataset = dataset['train'].train_test_split(test_size=0.20, seed=42)
    train_dataset = temp_dataset['train']
    train_dataset = train_dataset.map(add_id, with_indices=True)
    train_dataset = train_dataset.select_columns(['id', 'question', 'passage'])
    train_dataset = train_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "passage": "relevant_docs"
    })
    val_dataset = temp_dataset['test']
    val_dataset = val_dataset.map(add_id, with_indices=True)
    val_dataset = val_dataset.select_columns(['id', 'question', 'passage'])
    val_dataset = val_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "passage": "relevant_docs"
    })
    test_dataset = dataset['validation']
    test_dataset = test_dataset.map(add_id, with_indices=True)
    test_dataset = test_dataset.select_columns(['id', 'question', 'passage'])
    test_dataset = test_dataset.rename_columns({
        "id": "q_id", 
        "question": "question", 
        "passage": "relevant_docs"
    })
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_covid():
    
    def get_full_window(context, answer_start, window_size=150):
        start = max(0, answer_start - window_size)
        end = min(len(context), answer_start + window_size)
        return context[start:end]

    def add_context_window(example):
        answer_start = example['answers']['answer_start'][0]
        context = example['context']
        example['context_window'] = get_full_window(context, answer_start)
        return example
    
    DATASET_NAME = 'deepset/covid_qa_deepset'
    print("Loading and preparing datasets...")
    dataset= load_dataset(DATASET_NAME,split='train')
    dataset = dataset.map(add_context_window)
    temp_dataset = dataset.train_test_split(test_size=0.20, seed=42)
    dev_dataset = temp_dataset['train'].train_test_split(test_size=0.20, seed=42)
    train_dataset = dev_dataset['train']
    train_dataset = train_dataset.select_columns(['id', 'question', 'context_window'])
    train_dataset = train_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "context_window": "relevant_docs"
    })
    val_dataset = dev_dataset['test']
    val_dataset = val_dataset.select_columns(['id', 'question', 'context_window'])
    val_dataset = val_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "context_window": "relevant_docs"
    })
    test_dataset = temp_dataset['test']
    test_dataset = test_dataset.select_columns(['id', 'question', 'context_window'])
    test_dataset = test_dataset.rename_columns({
        "id": "q_id", 
        "question": "question", 
        "context_window": "relevant_docs"
    })
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset