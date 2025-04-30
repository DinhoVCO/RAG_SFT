from sentence_transformers.evaluation import InformationRetrievalEvaluator
from utils.get_data_evaluator import get_data_ir_evaluator


def create_evaluator_information_retrieval(dataset_name, dataset):
    queries, corpus, relevant_docs = get_data_ir_evaluator(dataset_name, dataset)
    print("Creating evaluator...")
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=False,
        batch_size=16,
        name="telecom-ir-eval",
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1],
        ndcg_at_k=[10],
        mrr_at_k=[3, 5, 10],
    )
    print("Evaluator created.")
    return evaluator