from string import Template
from utils.pre_processing_docs import remove_duplicates, create_text_splitter, split_documents
from transformers import (
    AutoTokenizer,
)

template_RAG = Template(
    "Instruct: You are given a context and a question. Answer only \"yes\" or \"no\" based solely on the information contained in the context. "
    "If the context does not provide enough information to answer confidently, answer based on the most likely interpretation from the given text.\n\n"
    "Context:\n$context\n\n"
    "Question:\n$question\n\n"
    "Output:"
)

template_base = Template(
    "Instruct: Answer the question based on your knowledge. Answer only \"yes\" or \"no\" .\n"
    "Question:\n$question\n\n"
    "Output:"
)

def format_input_context_boolq(row, context=None):
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

def get_full_promt_boolq(row, include_docs=True):
    def get_context(row):
        relevant_docs = row['relevant_documents']
        passage_gold = row['passage']
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
    question = format_input_context_boolq(row, context)
    answer = row['answer']
    return f"{question}\n{answer}"