import sys
sys.path.append('../../../src')
from vector_stores.faiss import VectorStoreFaiss
from transformers import pipeline
from tqdm import tqdm
from prompts.prompt_for_inference import prompt_for_inference
from datasets import Dataset


class RAGPipeline:
    def __init__(self, dataset_name, llm: pipeline, knowledge_index: VectorStoreFaiss = None):
        self.llm = llm
        self.knowledge_index = knowledge_index
        self.dataset_name = dataset_name

    def answer_batch(self, dataset:Dataset, retriever_bs=100, llm_bs = 20, num_retrieved_docs: int = 30):
        query_list = dataset['question']
        prompts=[]
        if(self.knowledge_index):
            relevant_docs = self.knowledge_index.buscar_por_batches(query_list, top_k=num_retrieved_docs, batch_size=retriever_bs)
        for i in tqdm(range(len(dataset)), desc="Generating prompts"):
            if(self.knowledge_index):
                context = "".join([f"\nDocument {str(j+1)}:" + doc for j, doc in enumerate(relevant_docs[i][0])])
                final_prompt = prompt_for_inference(self.dataset_name,dataset[i], context)
                prompts.append(final_prompt)
            else :
                final_prompt = prompt_for_inference(self.dataset_name,dataset[i])
                prompts.append(final_prompt)
        print("Example prompt")
        print(prompts[0])
        print("Processing inference")
        answers = self.llm(prompts, batch_size=llm_bs)
        return answers

    