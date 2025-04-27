import os
import sys
sys.path.append('../src')
import json
import argparse
from sentence_transformers import SentenceTransformer
from utils.data_for_train_emb import load_and_prepare_datasets
from utils.ir_evaluator import create_evaluator_information_retrieval


def parse_arguments():
    parser = argparse.ArgumentParser(description="Embedding model training")
    parser.add_argument('--name_dataset', type=str, required=True, help="Dataset name(clapnq, teleqna ")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for the results")
    parser.add_argument('--models_dir', type=str, required=True, help="Directory for the models")
    args = parser.parse_args()
    return args

def encontrar_checkpoint_unico(model_path):
    """Busca el único checkpoint dentro de la carpeta del modelo."""
    for d in os.listdir(model_path):
        dir_path = os.path.join(model_path, d)
        if os.path.isdir(dir_path) and d.startswith('checkpoint-'):
            return dir_path
    return None

def load_models(model_dir, model_names):
    print("Cargando modelos...")
    models = {}
    # Recorrer todos los archivos en la carpeta de modelos
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        if os.path.isdir(model_path):  # Verifica que sea un directorio de modelo
            checkpoint_dir = encontrar_checkpoint_unico(model_path)
            if checkpoint_dir:
            
                model_key = f"ft_{model_name}"
                models[model_key] = SentenceTransformer(checkpoint_dir)
            else:
                print(f"[ADVERTENCIA] No se encontró un checkpoint en {model_path}")

    for model_key, model_name in model_names.items():
        if model_key not in models:  # Evitar sobrescribir modelos ya cargados
            models[model_key] = SentenceTransformer(model_name)

    return models

# Modelos de lenguaje pre-entrenados
model_names = {
    'small': 'BAAI/bge-small-en-v1.5',
    'base': 'BAAI/bge-base-en-v1.5',
    'large': 'BAAI/bge-large-en-v1.5'
}


def main():
    args = parse_arguments()
    output_dir = args.output_dir
    models_dir = args.models_dir

    models = load_models(models_dir, model_names)
    print("Models loaded")
    print("Loading datasets...")
    # Evaluador
    train_dataset, val_dataset, test_dataset = load_and_prepare_datasets(args.name_dataset)
    print("Loaded dataset")
    # Evaluator
    evaluator = create_evaluator_information_retrieval(args.name_dataset, test_dataset)
    print("Evaluating models")

    print("Save results...")
    metrics = {}
    for model_key, model_to_evaluate in models.items():
        metrics[model_key] = evaluator(model_to_evaluate)

    # Guardar las métricas en formato JSON
    with open(f"{output_dir}/metrics_checkpoint_embedding.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()