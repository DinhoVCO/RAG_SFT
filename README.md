# 📡 Aprimorando RAG com Modelos Leves  

Os modelos de linguagem são eficazes em diversas tarefas de processamento de texto, mas enfrentam dificuldades em domínios especializados, como telecomunicações, devido à complexidade técnica e constante evolução dos padrões. Para solucionar esse problema, este estudo aprimora um sistema de **Recuperação e Geração Aumentada (RAG)** adaptado para responder perguntas sobre as especificações **3GPP**, um conjunto de normas fundamentais para redes móveis.  

A abordagem proposta utiliza **modelos leves** para equilibrar desempenho e eficiência computacional. O modelo **bge-small-en-v1.5** é ajustado para recuperar informações técnicas com maior precisão, enquanto o modelo **phi-2** passa por um fine-tuning para gerar respostas mais precisas e contextualizadas. Para otimizar esse processo, os documentos técnicos são segmentados estrategicamente e armazenados em um banco de dados vetorial **FAISS**, permitindo buscas eficientes. Além disso, um re-ranqueador baseado no modelo **ColBERT** refina a seleção dos documentos mais relevantes, e um índice especializado de abreviações do **3GPP** enriquece a compreensão do contexto técnico.  

Os experimentos demonstraram um **aumento de 22,38% na precisão das respostas**, tornando a solução escalável e viável para aplicações reais no setor de telecomunicações. Essa abordagem reduz os custos computacionais e possibilita a implementação em ambientes com recursos limitados. Como próximos passos, a pesquisa pretende expandir a base de conhecimento e aprimorar a estratégia de re-ranqueamento para continuar melhorando a precisão do sistema.  


![RAG Fine tuning](./paper/RAG_3gpp_FT.drawio.png)


## 📊 Conjuntos de Dados Utilizados  

- **[TeleQnA](https://huggingface.co/datasets/dinho1597/3GPP-QA-MultipleChoice)** → Conjunto com 10.000 perguntas sobre telecomunicações, categorizadas em léxico, pesquisa e especificações 3GPP.  
- CovidQA
- CLAPnq
- BoolQ

## Fast Start

O primeiro passo é baixar o código do repositório contendo os scripts necessários:  
```bash
!git clone https://github.com/DinhoVCO/RAG_FT_SLM.git
```
2️⃣ **Instalar as dependências**
```bash
!pip install -r /RAG_FT_SLM/requirements.txt 
```

## 🎯 Ajustando o Modelo de Embeddings  

Para realizar o **fine-tuning** do modelo de embeddings **bge-small-en-v1.5**, utilizamos um ambiente como **Google Colab** ou **Jupyter Notebook**. A seguir, apresentamos os passos necessários para configurar e treinar o modelo.  

###  Ajuste Fino do embedding

**Executar o script de ajuste fino**
```bash
!python  ../scripts/train_embedding.py \
  --name_dataset "boolq" \
  --model_name "BAAI/bge-small-en-v1.5" \
  --new_model_name "bge-small-boolq" \
  --epochs 10 \
  --batch_size 128 \
  --output_dir "../models/boolq/embedding/"
```
📌 Nota: Você pode modificar os parâmetros --epoch e --batch_size para ajustar o tempo de treinamento e o consumo de memória.

## 🏆 Avaliação do Modelo de Embeddings  

```bash
!python  ../scripts/evaluate_embedding.py \
  --name_dataset "boolq" \
  --output_dir "../results/boolq/" \
  --models_dir "../models/boolq/embedding/"
```

## Index FAISS 
```bash
!python ../scripts/create_vector_store.py \
  --dataset "boolq" \
  --emb_model "../models/boolq/embedding/bge-small-boolq_10e_128bs" \
  --cs 150 \
  --co 20 \
  --bs_emb 1024 \
  --output_dir "../vector_stores/boolq/ft_"
```

## Ajuste Fino Lora phi-2
```bash
!python  ../scripts/ft_phi.py \
  --new_model_name "phi_2_rag_k1_boolq_2e_10bs" \
  --num_epochs 2 \
  --batch_size 10 \
  --dataset_name "boolq" \
  --include_docs \
  --top_k 1 \
  --save_path "../models/boolq/adapters/" \
  --vector_store_path "../vector_stores/boolq/ft_vs_boolq_150_20"
```

## RAG Inference 

```bash
!python ../scripts/inference_rag.py \
  --lora_adapter_path "../models/boolq/adapters/best_phi_2_rag_k1_boolq_2e_10bs" \
  --max_new_tokens 10 \
  --vector_store_path "../vector_stores/boolq/ft_vs_boolq_150_20" \
  --dataset_name "boolq" \
  --output_csv_path "../results/boolq/full_ft_k10_boolq.csv" \
  --bs_emb 50 \
  --bs_gen 8 \
  --top_k 10 \
  --use_rag
```
### Só phi-2

```bash
!python ../scripts/inference_rag.py \
  --gen_model "microsoft/phi-2" \
  --max_new_tokens 3 \
  --dataset_name "boolq" \
  --output_csv_path "../results/boolq/phi_k10_boolq.csv" \
  --bs_emb 10 \
  --bs_gen 10\
  --top_k 10
```

## Evaluar

```python
from utils.evaluate_inference import evaluate_answer
evaluate_answer('boolq', '../results/boolq/full_ft_k10_boolq.csv')
```
