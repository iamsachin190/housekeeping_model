
import os
from typing import List, Any
from PIL import Image

# Fix for old SQLite versions on server (ChromaDB requires SQLite > 3.35)
try:
    import sys
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from app.config import settings

class CLIPEmbeddingFunction(EmbeddingFunction):
    """
    Singleton wrapper for CLIP embeddings to prevent reloading model on every request.
    """
    def __init__(self):
        # Using a lightweight CLIP model
        token = settings.HF_TOKEN if settings.HF_TOKEN else None
        self.model = SentenceTransformer('clip-ViT-B-32', token=token)

    def __call__(self, input: Any) -> List[List[float]]:
        batch = []
        for item in input:
            if isinstance(item, str) and os.path.exists(item):
                try:
                    batch.append(Image.open(item))
                except Exception:
                    batch.append(item) # Fallback to text if image load fails
            else:
                batch.append(item) # Handle text input

        embeddings = self.model.encode(batch)
        return embeddings.tolist()

# Initialize Global Instances
_chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
_embedding_fn = CLIPEmbeddingFunction()

try:
    collection = _chroma_client.get_collection(name="bims_examples", embedding_function=_embedding_fn)
except Exception:
    collection = _chroma_client.create_collection(name="bims_examples", embedding_function=_embedding_fn)

def add_to_index(image_path: str, status: str, description: str):
    """Adds a reference image to the vector DB."""
    import uuid
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[image_path], 
        metadatas=[{"status": status, "description": description}]
    )

def retrieve_similar_context(query_image: Image.Image) -> str:
    """
    Encodes the query image and finds similar 'Known' examples.
    """
    # Encode directly using the embedding function's model
    query_embedding = _embedding_fn.model.encode([query_image]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    
    context_str = "Reference Examples found in database:\n"
    if not results['metadatas'] or not results['metadatas'][0]:
        return context_str + "No reference images found."

    for i, meta in enumerate(results['metadatas'][0]):
        context_str += f"- Example {i+1}: Status={meta['status']}, Note={meta['description']}\n"
        
    return context_str
