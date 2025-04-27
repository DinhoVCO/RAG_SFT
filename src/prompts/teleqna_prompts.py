import os
import sys
from string import Template
import pandas as pd
import random

template_RAG = Template(
    "Instruct: Use the context provided to select the correct option. Select the correct option from $valid_options. Respond only with the letter of the correct option.\n"
    "Context:\n$explanation\n\n"
    "Question:\n$question\n\n"
    "Options:\n$options\n\n"
    "Output:"
)

template_base = Template(
    "Instruct: Answer the following question by selecting the correct option. Select the correct option from $valid_options. Respond only with the letter of the correct option.\n"
    "Question:\n$question\n\n"
    "Options:\n$options\n\n"
    "Output:"
)

def format_input_context_teleqna(row, context=None, abbreviations=None):
    question_text = row['question'] 
    options_dict = {
        'A': row['A'],
        'B': row['B'],
        'C': row['C'],
        'D': row['D'],
        'E': row['E'],
    }
    valid_options = [key for key, value in options_dict.items() if pd.notna(value) and value is not None]
    valid_options_text = ", ".join(valid_options)
    options_text = "\n".join([f"{key}) {value}" for key, value in options_dict.items() if key in valid_options])
    if context:
        input_text = template_RAG.substitute(
            valid_options=valid_options_text,
            question=question_text,
            options=options_text,
            explanation=context
        )
    else:
        input_text = template_base.substitute(
            valid_options=valid_options_text,
            question=question_text,
            options=options_text
        )

    return input_text, valid_options

def get_full_promt_teleqna(row, include_docs=True):
    def get_answer(row):
        ans = row['answer']
        full_ans = row[ans]
        return f"{ans}) {full_ans}"

    def get_context(row):
        relevant_docs = row['relevant_documents']
        posicion_aleatoria = random.randint(0, len(relevant_docs))
        relevant_docs.insert(posicion_aleatoria, row['explanation'])
        context = ""
        context += "".join(
            [f"\nDocument {str(i)}:" + doc for i, doc in enumerate(relevant_docs)]
        )
        return context

    context = None
    if(include_docs):
        context = get_context(row)
    question = format_input_context_teleqna(row, context)[0]
    answer = get_answer(row)
    return f"{question}\n{answer}"