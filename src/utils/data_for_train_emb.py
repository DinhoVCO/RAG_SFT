from datasets import load_dataset
from utils.datasets_splits import load_dataset_splits

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
    train_dataset, val_dataset, test_dataset = load_dataset_splits('teleqna')
    train_dataset = train_dataset.select_columns(['question_id', 'question', 'explanation'])
    train_dataset = train_dataset.rename_columns({
        "question_id": "q_id",
        "question": "question",
        "explanation": "relevant_docs"
    })
    val_dataset = val_dataset.select_columns(['question_id', 'question', 'explanation'])
    val_dataset = val_dataset.rename_columns({
        "question_id": "q_id",
        "question": "question",
        "explanation": "relevant_docs"
    })
    test_dataset = test_dataset.select_columns(['question_id', 'question', 'explanation'])
    test_dataset = test_dataset.rename_columns({
        "question_id": "q_id", 
        "question": "question", 
        "explanation": "relevant_docs"
    })
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset

def get_passages_for_squad():
    train_dataset, val_dataset, test_dataset = load_dataset_splits('squad')
    train_dataset = train_dataset.select_columns(['id', 'question', 'context'])
    train_dataset = train_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "context": "relevant_docs"
    })
    val_dataset = val_dataset.select_columns(['id', 'question', 'context'])
    val_dataset = val_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "context": "relevant_docs"
    })
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
    
    train_dataset, val_dataset, test_dataset = load_dataset_splits('clapnq')
    #only answerable rows
    #train_dataset = train_dataset.filter(lambda row: row['output'][0]['answer'] != '', load_from_cache_file=False)
    #val_dataset = val_dataset.filter(lambda row: row['output'][0]['answer'] != '', load_from_cache_file=False)
    #test_dataset = test_dataset.filter(lambda row: row['output'][0]['answer'] != '', load_from_cache_file=False)


    train_dataset = train_dataset.map(extract_title_and_text, load_from_cache_file=False)    
    train_dataset = train_dataset.select_columns(['id', 'input', 'context'])
    train_dataset = train_dataset.rename_columns({
        "id": "q_id",
        "input": "question",
        "context": "relevant_docs"
    })
    val_dataset = val_dataset.map(extract_title_and_text, load_from_cache_file=False)
    val_dataset = val_dataset.select_columns(['id', 'input', 'context'])
    val_dataset = val_dataset.rename_columns({
        "id": "q_id",
        "input": "question",
        "context": "relevant_docs"
    })
    test_dataset = test_dataset.map(extract_title_and_text, load_from_cache_file=False)
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

    train_dataset, val_dataset, test_dataset = load_dataset_splits('boolq')
    train_dataset = train_dataset.map(add_id, with_indices=True, load_from_cache_file=False)
    train_dataset = train_dataset.select_columns(['id', 'question', 'passage'])
    train_dataset = train_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "passage": "relevant_docs"
    })
    val_dataset = val_dataset.map(add_id, with_indices=True, load_from_cache_file=False)
    val_dataset = val_dataset.select_columns(['id', 'question', 'passage'])
    val_dataset = val_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "passage": "relevant_docs"
    })
    test_dataset = test_dataset.map(add_id, with_indices=True, load_from_cache_file=False)
    test_dataset = test_dataset.select_columns(['id', 'question', 'passage'])
    test_dataset = test_dataset.rename_columns({
        "id": "q_id", 
        "question": "question", 
        "passage": "relevant_docs"
    })
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset


def get_passages_for_covid():
    
    def get_full_window(context, answer_start, window_size=220):
        start = max(0, answer_start - window_size)
        end = min(len(context), answer_start + window_size)
        return context[start:end]

    def add_context_window(example):
        answer_start = example['answers']['answer_start'][0]
        context = example['context']
        example['context_window'] = get_full_window(context, answer_start)
        return example
    
    train_dataset, val_dataset, test_dataset = load_dataset_splits('covid')
    
    train_dataset = train_dataset.map(add_context_window, load_from_cache_file=False)
    val_dataset = val_dataset.map(add_context_window, load_from_cache_file=False)
    test_dataset = test_dataset.map(add_context_window, load_from_cache_file=False)
    
    train_dataset = train_dataset.select_columns(['id', 'question', 'context_window'])
    train_dataset = train_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "context_window": "relevant_docs"
    })
    val_dataset = val_dataset.select_columns(['id', 'question', 'context_window'])
    val_dataset = val_dataset.rename_columns({
        "id": "q_id",
        "question": "question",
        "context_window": "relevant_docs"
    })
    test_dataset = test_dataset.select_columns(['id', 'question', 'context_window'])
    test_dataset = test_dataset.rename_columns({
        "id": "q_id", 
        "question": "question", 
        "context_window": "relevant_docs"
    })
    print("Datasets loaded and prepared.")
    return train_dataset, val_dataset, test_dataset