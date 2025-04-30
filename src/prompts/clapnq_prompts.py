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
    "Instruct: Using the information in the context, answer the question as concisely and faithfully as possible. If the context does not contain enough information,  respond with unanswerable.\n"
    "Context:\n$context\n\n"
    "Question:\n$question\n\n"
    "Output:"
)

template_base = Template(
    "Instruct: Only answer the question if you are certain based on your knowledge. Otherwise, respond with unanswerable.\n"
    "Question:\n$question\n\n"
    "Output:"
)

def format_input_context_clapnq(row, context=None):
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

def get_full_promt_clapnq(row, include_docs=True):
    def extract_title_and_text(example):
        if isinstance(example['passages'], list) and len(example['passages']) > 0:
            entry = example['passages'][0]
            title = entry.get('title', '')
            text = entry.get('text', '')
            return f"{title}, {text}"
        return ''
    def get_answer(row):
        ans = row['output'][0]['answer']
        if ans=='':
            return 'unanswerable'
        else:
            return ans        

    def get_context(row):
        relevant_docs = row['relevant_documents']
        passage_gold = extract_title_and_text(row)
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        text_splitter = create_text_splitter(150, 20, tokenizer)
        doc_passages = text_splitter.split_text(passage_gold)
        relevant_docs += doc_passages[:3]
        context = ""
        context += "".join(
            [f"\nDocument {str(i)}:" + doc for i, doc in enumerate(relevant_docs)]
        )
        return context

    context = None
    if(include_docs):
        context = get_context(row)
    question = format_input_context_clapnq(row, context)
    answer = get_answer(row)
    return f"{question}\n{answer}"