import sys
import os
from utils.datasets_splits import load_dataset_splits
import pandas as pd
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from ragas.metrics import ExactMatch
from datasets import Dataset

def evaluate_answer(dataset_name, path_inference):
    train_ds, val_ds, test_ds = load_dataset_splits(dataset_name)
    if(dataset_name=='covid'):
        evaluate_answer_covid(test_ds, path_inference)
    elif(dataset_name=='clapnq'):
        evaluate_answer_clapnq(test_ds, path_inference)
    elif(dataset_name=='teleqna'):
        evaluate_answer_teleqna(test_ds, path_inference)
    elif(dataset_name=='boolq'):
        evaluate_answer_boolq(test_ds, path_inference)
    else:
        print("Error name dataset")

def get_bleu(inference_answers, true_answers):
    bleu = BLEU()
    refs = [[ref] for ref in true_answers]
    bleu_corpus = bleu.corpus_score(inference_answers, refs)


    score = bleu_corpus.score

    return score


def get_rouge(inference_answers, true_answers):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [
        scorer.score(pred, ref)['rougeL'].fmeasure
        for pred, ref in zip(inference_answers, true_answers)
    ]
    avg_score = sum(scores) / len(scores)
    return scores, avg_score

def save_final_result(path_inference, avg_score, bleu_corpus, accuracy):
    filename = os.path.basename(path_inference)
    summary_df = pd.DataFrame([[filename, round(avg_score, 2), round(bleu_corpus, 2), accuracy]], columns=["archivo_csv", "rougeL_f1", "bleu_corpus", "accuracy"])
    summary_path = "../results/final_results.csv"
    if not os.path.exists(summary_path):
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)

def get_accuracy(inference_answers, true_answers):
    if len(inference_answers) != len(true_answers):
        raise ValueError("Las listas de respuestas deben tener la misma longitud.")
    
    correct_count = sum(1 for i, answer in enumerate(inference_answers) if answer.strip().lower() == true_answers[i])
    accuracy = correct_count / len(true_answers) * 100 
    return accuracy
    

def evaluate_answer_covid(test_ds, path_inference):
    def get_true_answers(row):
        answer = row['answers']['text'][0]  
        return {'answer': answer}

    df = pd.read_csv(path_inference)
    inference_answers = df["inference"].tolist()
    test_ds = test_ds.select_columns(['id','question','answers'])  
    test_ds = test_ds.map(get_true_answers)
    true_answers = test_ds['answer']
    
    scores_rouge, avg_score = get_rouge(inference_answers, true_answers)
    
    df["question"] = test_ds['question']
    df["id"] = test_ds['id']
    df["true_answer"] = true_answers
    df = df[['id', 'question', 'true_answer', 'inference']]
    df.to_csv(path_inference, index=False)
    save_final_result(path_inference, avg_score, 0, 0)
    print(f"Results saved in: {path_inference}")

def evaluate_answer_clapnq(test_ds, path_inference):
    def get_true_answers(row):
        answer = row['output'][0]['answer']
        if(answer==''):
            return {'answer': 'unanswerable'}
        return {'answer': answer}

    df = pd.read_csv(path_inference)
    inference_answers = df["inference"].tolist()
    test_ds = test_ds.select_columns(['id','input','output'])  
    test_ds = test_ds.map(get_true_answers)
    
    scores_rouge, avg_score = get_rouge(inference_answers[:300], test_ds['answer'][:300])
    #bleu_corpus = get_bleu(inference_answers[:300], test_ds['answer'][:300])    

    accuracy = get_accuracy(inference_answers[300:], test_ds['answer'][300:])
    
    df["question"] = test_ds['input']
    df["id"] = test_ds['id']
    df["true_answer"] = test_ds['answer']
    df = df[['id', 'question', 'true_answer', 'inference']]
    df.to_csv(path_inference, index=False)
    save_final_result(path_inference, avg_score, 0, accuracy)
    print(f"Results saved in: {path_inference}")

def get_accuracy_teleqna(inference_answers, true_answers):
    if len(inference_answers) != len(true_answers):
        raise ValueError("Las listas de respuestas deben tener la misma longitud.")
    correct_count = sum(1 for i, answer in enumerate(inference_answers) if answer.strip() == true_answers[i])
    accuracy = correct_count / len(true_answers) * 100 
    return accuracy


def evaluate_answer_teleqna(test_ds, path_inference):
    def get_true_answers(row):
        option = row['answer']
        full_ans = row[option]
        return {'true_answer': f"{option}) {full_ans}"}

    df = pd.read_csv(path_inference)
    inference_answers = df["inference"].tolist()
    answer_options = [opcion.replace("\n", "").strip()[0] for opcion in inference_answers]
    test_ds = test_ds.map(get_true_answers)
    true_answers = test_ds['true_answer']
    
    scores_rouge, avg_score = get_rouge(inference_answers, true_answers)
    accuracy = get_accuracy_teleqna(answer_options, test_ds['answer'])
    
    df["question"] = test_ds['question']
    df["id"] = test_ds['question_id']
    df["true_answer"] = true_answers
    df = df[['id', 'question', 'true_answer', 'inference']]
    df.to_csv(path_inference, index=False)
    save_final_result(path_inference, avg_score, 0, accuracy)
    print(f"Results saved in: {path_inference}")

def evaluate_answer_boolq(test_ds, path_inference):
    def get_accuracy_boolq(inference_answers, true_answers):
        if len(inference_answers) != len(true_answers):
            raise ValueError("Las listas de respuestas deben tener la misma longitud.")
        
        correct_count = sum(1 for i, answer in enumerate(inference_answers) if answer.strip() == true_answers[i])
        accuracy = correct_count / len(true_answers) * 100 
        return accuracy
    df = pd.read_csv(path_inference)
    inference_answers = df["inference"].tolist()
    test_ds = test_ds.select_columns(['question','answer'])  
    true_answers = test_ds['answer']
    yes_no = ["True" if b else "False" for b in true_answers]
    accuracy = get_accuracy_boolq(inference_answers, yes_no)    
    df["question"] = test_ds['question']
    df["true_answer"] = yes_no
    df = df[['question', 'true_answer', 'inference']]
    df.to_csv(path_inference, index=False)
    save_final_result(path_inference, 0, 0, accuracy)
    print(f"Results saved in: {path_inference}")
    
