import os
import sys
from string import Template
import pandas as pd
import random
from utils.pre_processing_docs import remove_duplicates, create_text_splitter, split_documents
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

template_RAG = Template(
    "Instruct:  Using the information in the context, answer the question as concisely and faithfully as possible."
    "If the context does not provide enough information to answer confidently, answer based on the most likely interpretation from the given text.\n\n"
    "Context:\n$context\n\n"
    "Question:\n$question\n\n"
    "Output:"
)

template_base = Template(
    "Instruct: Answer the question based on your knowledge.\n"
    "Question:\n$question\n\n"
    "Output:"
)

def format_input_context_covid(row, context=None):
    question_text = row['question'] 
    if context:
        input_text = template_RAG.substitute(
            question=question_text,
            context=context
        )
    else:
        input_text = template_base.substitute(
            question=question_text,
        )

    return input_text

def get_full_promt_covid(row, include_docs=True):
    
    def get_full_window(context, answer_start, window_size=150):
        start = max(0, answer_start - window_size)
        end = min(len(context), answer_start + window_size)
        return context[start:end]
    
    def get_context(row):
        relevant_docs = row['relevant_documents']
        answer_start = row['answers']['answer_start'][0]
        passage_gold = get_full_window(row['context'], answer_start)
        posicion_aleatoria = random.randint(0, len(relevant_docs))
        relevant_docs.insert(posicion_aleatoria, passage_gold)
        context = ""
        context += "".join(
            [f"\nDocument {str(i)}:" + doc for i, doc in enumerate(relevant_docs)]
        )
        return context

    context = None
    if(include_docs):
        context = get_context(row)
    question = format_input_context_covid(row, context)
    answer = row['answers']
    answer_text=answer['text'][0]
    return f"{question}\n{answer_text}"