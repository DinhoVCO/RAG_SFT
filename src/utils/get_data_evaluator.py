def get_data_ir_evaluator(dataset_name, dataset):
    if(dataset_name=="teleqna"):
        return ir_data_for_teleqna(dataset)
    if(dataset_name=="squad"):
        return ir_data_for_squad(dataset)
    if(dataset_name=="boolq"):
        return ir_data_for_boolq(dataset)
    if(dataset_name=="clapnq"):
        return ir_data_for_clapnq(dataset)
    if(dataset_name=="covid"):
        return ir_data_for_covid(dataset)
    else:
        raise ValueError(f"Incorrect dataset name: {dataset_name}")

def ir_data_for_clapnq(dataset):
    queries = {}
    corpus = {}
    relevant_docs = {}
    context_to_id = {}
    context_id_counter = 0

    for example in dataset:
        q_id = example['q_id']
        question = example['question']
        context = example['relevant_docs']

        # Si el contexto ya existe, usamos su ID
        if context not in context_to_id:
            context_id = f"c{context_id_counter}"  # ejemplo: c0, c1, c2, ...
            context_to_id[context] = context_id
            corpus[context_id] = context
            context_id_counter += 1
        else:
            context_id = context_to_id[context]

        queries[q_id] = question
        relevant_docs[q_id] = [context_id]

    return queries, corpus, relevant_docs
    
def ir_data_for_boolq(dataset):
    corpus = dict(zip(dataset["q_id"], dataset["relevant_docs"]))
    queries = dict(zip(dataset["q_id"], dataset["question"]))
    relevant_docs = {}
    for q_id in queries:
        relevant_docs[q_id] = [q_id]
    return queries, corpus, relevant_docs
    
def ir_data_for_covid(dataset):
    queries = {}
    corpus = {}
    relevant_docs = {}
    context_to_id = {}
    context_id_counter = 0

    for example in dataset:
        q_id = example['q_id']
        question = example['question']
        context = example['relevant_docs']

        # Si el contexto ya existe, usamos su ID
        if context not in context_to_id:
            context_id = f"c{context_id_counter}"  # ejemplo: c0, c1, c2, ...
            context_to_id[context] = context_id
            corpus[context_id] = context
            context_id_counter += 1
        else:
            context_id = context_to_id[context]

        queries[q_id] = question
        relevant_docs[q_id] = [context_id]

    return queries, corpus, relevant_docs

def ir_data_for_teleqna(dataset):
    corpus = dict(zip(dataset["q_id"], dataset["relevant_docs"]))
    queries = dict(zip(dataset["q_id"], dataset["question"]))
    relevant_docs = {}
    for q_id in queries:
        relevant_docs[q_id] = [q_id]
    return queries, corpus, relevant_docs

def ir_data_for_squad(dataset):
    queries = {}
    corpus = {}
    relevant_docs = {}
    context_to_id = {}
    context_id_counter = 0

    for example in dataset:
        q_id = example['q_id']
        question = example['question']
        context = example['relevant_docs']

        # Si el contexto ya existe, usamos su ID
        if context not in context_to_id:
            context_id = f"c{context_id_counter}"  # ejemplo: c0, c1, c2, ...
            context_to_id[context] = context_id
            corpus[context_id] = context
            context_id_counter += 1
        else:
            context_id = context_to_id[context]

        queries[q_id] = question
        relevant_docs[q_id] = [context_id]

    return queries, corpus, relevant_docs