import faiss
import numpy as np
from utils.save_and_load import save_docs_to_jsonl, load_docs_from_jsonl
from tqdm import tqdm
import torch
import gc
import os

class VectorStoreFaiss:
    def __init__(self, embedding_model, index_type = faiss.IndexFlatIP):
        self.embedding_model = embedding_model
        self.index = None
        self.embeddings = None
        self.documents = None
        self.index_type = index_type

    def embeddings_batch_gpu(self, documents, batch_size=1):
        """
        Genera embeddings directamente en la GPU usando procesamiento por batches.
        """
        embeddings_totales = []
        num_batches = len(documents) // batch_size + int(len(documents) % batch_size != 0)

        for i in tqdm(range(0, len(documents), batch_size), desc="Generando embeddings", total=num_batches):
            batch = documents[i : i + batch_size]
            textos_batch = [doc.page_content for doc in batch]
            with torch.no_grad():
                embeddings = self.embedding_model.encode(
                    textos_batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            embeddings_totales.append(embeddings)

        self.embeddings = np.vstack(embeddings_totales)
        self.documents = documents

    def crear_indice_faiss(self):
        """
        Crea un índice FAISS usando los embeddings generados.
        """
        if self.embeddings is None:
            raise ValueError("No hay embeddings generados. Ejecuta 'generar_embeddings_batch_gpu' primero.")
        
        dimension = self.embeddings.shape[1]
        self.index = self.index_type(dimension)
        self.index.add(self.embeddings)
        print("\u2705 Índice FAISS creado exitosamente.")

    def buscar(self, query, top_k=5):
        """
        Realiza una búsqueda en el índice dado un query.
        """
        if self.index is None:
            raise ValueError("El índice FAISS no ha sido creado.")
        
        with torch.no_grad():
            query_embedding = self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False
            )
        
        D, I = self.index.search(query_embedding, top_k)
        resultados = [self.documents[idx].page_content for idx in I[0]]
        return resultados, D[0]

    def buscar_por_batches(self, query_list, top_k=5, batch_size=1):
        resultados_totales = []
        for i in tqdm(range(0, len(query_list), batch_size), desc="🔍 Buscando"):
            batch = query_list[i : i + batch_size]
            with torch.no_grad():
                embeddings_batch = self.embedding_model.encode(
                    batch,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
            distancias, indices = self.index.search(embeddings_batch, top_k)

            for j, consulta in enumerate(batch):
                resultados = [
                    self.documents[int(idx)].page_content for k, idx in enumerate(indices[j])
                ]
                resultados_totales.append((resultados, distancias[j]))
        return resultados_totales

    def save_local(self, folder_path):
        """
        Guarda el índice FAISS a un archivo.
        """
        if self.index is None:
            raise ValueError("El índice FAISS no ha sido creado.")
        # Crear carpeta si no existe
        if folder_path != "":  # Por si el path es solo un nombre de archivo
            os.makedirs(folder_path, exist_ok=True)

        faiss.write_index(self.index, folder_path+"/faiss_index")
        save_docs_to_jsonl(self.documents, folder_path+"/documents.jsonl")
        print(f"💾 Vector store guardado en {folder_path}")

    @classmethod
    def load_local(cls, embedding_model, folder_path):
        """
        Carga un índice FAISS desde un archivo.
        """
        index = faiss.read_index(folder_path+"/faiss_index")
        documents = load_docs_from_jsonl(folder_path+'/documents.jsonl')
        instance = cls(embedding_model)
        instance.index = index
        instance.documents = documents
        print(f"💾  Vector store cargado desde {folder_path}")
        return instance
        
    @classmethod
    def from_documents(cls, embedding_model, documents, batch_size=1, index_type=faiss.IndexFlatIP):
        """
        Clase método para crear una instancia de VectorStoreFaiss directamente desde documentos.
        """
        instancia = cls(embedding_model, index_type=index_type)
        instancia.embeddings_batch_gpu(documents, batch_size)
        instancia.crear_indice_faiss()
        # Limpieza automática de memoria
        del instancia.embeddings
        torch.cuda.empty_cache()
        gc.collect()
        return instancia
