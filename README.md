# 📡 Aprimorando Geração Aumentada por Recuperação via Ajuste Fino Sequencial de Modelos de Linguagem Pequenos 

Modelos de linguagem (Language Models, LMs) se destacam em conhecimento geral, mas frequentemente enfrentam dificuldades em domínios especializados, nos quais a complexidade e a constante evolução representam desafios adicionais. Este estudo visa aprimorar a efetividade de sistemas de Geração Aumentada por Recuperação (Retrieval-Augmented Generation, RAG) para a tarefa de Perguntas e Respostas (Question Answering, QA) por meio do ajuste sequencial dos componentes do RAG, utilizando modelos de linguagem pequenos (Small Language Models, SLMs). Nossa abordagem ajusta tanto o modelo de embedding quanto o modelo generativo utilizando poucos recursos computacionais e melhora a efetividade geral em relação ao vanilla RAG. A metodologia proposta, escalável e econômica, viabiliza a aplicação prática de sistemas RAG em diferentes domínios e tarefas.

![RAG Fine tuning](./images/rag_full_FT.png)

### Primeira fase : FT Embedding model
![Embedding Fine tuning](./images/first_step.png)

### Segunda fase :FT Generative model
![Generative model Fine tuning](./images/second_step.png)

## 📊 Conjuntos de Dados Utilizados  

- **[3GPP-QA-MultipleChoice](https://huggingface.co/datasets/DinoStackAI/3GPP-QA-MultipleChoice)**, subconjunto de [TeleQnA](https://huggingface.co/datasets/netop/TeleQnA) com perguntas sobre telecomunicações, categorizadas em léxico e especificações do 3GPP.  
- **[CovidQA](https://huggingface.co/datasets/deepset/covid_qa_deepset)**
- **[CLAPnq](https://huggingface.co/datasets/PrimeQA/clapnq)**
- **[BoolQ](https://huggingface.co/datasets/google/boolq)**

## Corpus
Os conjuntos de dados BoolQ e COVID-QA incluem corpus.
- **[3GPP-Release 18](https://huggingface.co/datasets/netop/3GPP-R18)** : Baixar o [arquivo .json](https://drive.google.com/file/d/1yX9GSqY-O31ruuLp1HRTvMPxzFwFbHOg/view?usp=sharing) e salvá-lo na pasta datasets/teleqna/corpus/
- **[CLAPnq](https://huggingface.co/datasets/PrimeQA/clapnq_passages)**


## Início Rápido
OBS: Recomendamos criar primeiro um ambiente virtual venv
```bash
python3.10 -m venv rag_ft
```
Ativar o entorno para Linux
```bash
source rag_ft/bin/activate
```
Ativar o entorno para Windows
```bash
rag_ft\Scripts\activate
```


1. O primeiro passo é baixar o código do repositório contendo os scripts necessários:  
```bash
git clone https://github.com/DinhoVCO/RAG_FT_SLM.git
```
2. **Instalar as dependências**
```bash
pip install -r /requirements.txt 
```
3.  **Instalar nossa biblioteca**
```bash
pip install -e .
```
4. Para trabalhar com Jupyter Notebooks, é necessário criar um kernel:
```bash
python -m ipykernel install --user --name rag_ft --display-name "Python 3.10 (rag_ft)"
```
5. Alterar o nome do arquivo env para .env e preenchê-lo com as respectivas variáveis de ambiente.
```
AZURE_OPENAI_API_KEY = 
SSL_CERT_FILE2 = 
AZURE_OPENAI_ENDPOINT = 
OPENAI_API_VERSION = 2023-08-01-preview
MODELS = {"gpt-4o": "change_name_model", "gpt-3.5-turbo": "change_name_model"}
HUGGINGFACE_TOKEN=
WANDB_API_KEY=
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
📌 Nota: --name_dataset: teleqna, covid, clapnq, boolq.

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

## RAG Adapter Inference 

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
### Só phi-2 sem Retriever

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

## Notebooks
A pasta [notebooks](./notebooks/) contém notebooks de fine-tuning para cada conjunto de dados.

## Experimentos
A pasta [experiments](./experiments/) contém as diferentes configurações para avaliar nossa proposta.

## Report WanDB
 [WanDB-Report](https://api.wandb.ai/links/dinho15971-unicamp/f0nda1xb)

## Results 
 [resultados dos experimentos](https://drive.google.com/drive/folders/1iK9D_WIscWMUyBezEpgz_G2DPn4T89EK?usp=sharing)

## Como Citar
Se você encontrar este trabalho útil para sua pesquisa, por favor, considere citar nosso artigo:
```bibtex
@INPROCEEDINGS{247070,
    AUTHOR="Ronaldinho Vega Centeno Olivera and Julio Dos Reis and Frances Santos and Allan M. de Souza",
    TITLE="Aprimorando Geração Aumentada por Recuperação via Ajuste Fino Sequencial de Modelos de Linguagem Pequenos",
    BOOKTITLE="SBBD 2025 - Full Papers () ",
    ADDRESS="Fortaleza, CE, Brazil",
    DAYS="29-2",
    MONTH="sep",
    YEAR="2025",
    ABSTRACT="Modelos de linguagem (Language Models, LMs) se destacam em conhecimento geral, mas frequentemente enfrentam dificuldades em domínios especializados, nos quais a complexidade e a constante evolução representam desafios adicionais. Este estudo aprimora o desempenho de sistemas de Geração Aumentada por Recuperação (Retrieval-Augmented Generation, RAG) para a tarefa de Perguntas e Respostas (Question Answering, QA) por meio do ajuste sequencial dos componentes do RAG, utilizando modelos de linguagem pequenos (Small Language Models, SLMs) tanto para o modelo de embedding quanto para o modelo generativo. Nossa abordagem utiliza poucos recursos computacionais e melhora a efetividade geral em relação ao modelo base. A metodologia proposta, escalável e econômica, viabiliza a aplicação prática de sistemas RAG em diferentes domínios e tarefas.",
    KEYWORDS="Information retrieval; Machine Learning, AI, data management and data systems; Specialized and domain-specific data management; Text mining and natural language processing",
    URL="http://XXXXX/247070.pdf"
}
