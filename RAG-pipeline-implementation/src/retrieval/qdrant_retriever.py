# src/retrieval/qdrant_retriever.py
import os
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class QdrantRetriever:
    def __init__(self, collection_name="multimodal_rag", meta_file=None):
        """
        Initialize the Qdrant retriever.
        
        Args:
            collection_name (str): Name of the collection to use in Qdrant
            meta_file (str): Path to metadata JSON file (used for local metadata storage)
        """
        # Initialize Qdrant client with API key from environment variables
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "https://c1614c85-7e2f-4977-ad49-22100cf4ca32.eu-central-1-0.aws.cloud.qdrant.io:6333"),
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.ebDo7POtBL7HZDx6uBBjF4uhRz_KTRFDadn3YlXii_I"
        )
        self.collection_name = collection_name
        
        # Load metadata from file if provided (for compatibility with existing setup)
        self.meta = []
        if meta_file:
            try:
                with open(meta_file) as f:
                    self.meta = json.load(f)
            except Exception as e:
                print(f"Could not load metadata file: {e}")

    def create_collection(self, vector_size=512):
        """
        Create a new collection in Qdrant.
        
        Args:
            vector_size (int): Dimensionality of vectors to store
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create collection if it doesn't exist
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Created collection {self.collection_name} in Qdrant")
            else:
                print(f"Collection {self.collection_name} already exists in Qdrant")
        except Exception as e:
            print(f"Error creating Qdrant collection: {e}")

    def upload_vectors(self, vectors, metadata_list, batch_size=100):
        """
        Upload vectors and metadata to Qdrant.
        
        Args:
            vectors (np.ndarray): Array of vectors to upload
            metadata_list (list): List of dictionaries containing metadata
            batch_size (int): Number of vectors to upload per batch
        """
        try:
            # Create collection if it doesn't exist
            self.create_collection(vector_size=vectors.shape[1])
            
            # Prepare points for upload
            points = []
            for i, (vector, metadata) in enumerate(zip(vectors, metadata_list)):
                points.append(
                    models.PointStruct(
                        id=i,
                        vector=vector.tolist(),
                        payload=metadata
                    )
                )
            
            # Upload in batches
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            print(f"Uploaded {len(vectors)} vectors to Qdrant collection {self.collection_name}")
            self.meta = metadata_list
        except Exception as e:
            print(f"Error uploading vectors to Qdrant: {e}")

    def query(self, q_vec, top_k=5):
        """
        Query the Qdrant collection.
        
        Args:
            q_vec (np.ndarray): Query vector
            top_k (int): Number of results to return
            
        Returns:
            list: List of dictionaries containing search results with metadata
        """
        try:
            # Normalize query vector if needed
            q_vec = q_vec / np.linalg.norm(q_vec, axis=1, keepdims=True)
            
            # Search Qdrant
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=q_vec[0].tolist(),
                limit=top_k
            )
            
            # Format results
            out = []
            for hit in results:
                result = hit.payload.copy()
                result["score"] = float(hit.score)
                out.append(result)
            
            return out
        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            
            # Fallback to local metadata if Qdrant query fails
            if self.meta:
                print("Using local metadata as fallback")
                return self.meta[:min(top_k, len(self.meta))]
            return []

# Helper function to convert FAISS index to Qdrant collection
def upload_faiss_to_qdrant(faiss_index_path, meta_file_path, collection_name="multimodal_rag"):
    """
    Upload vectors from FAISS index to Qdrant.
    
    Args:
        faiss_index_path (str): Path to FAISS index
        meta_file_path (str): Path to metadata JSON file
        collection_name (str): Name of collection to create in Qdrant
    """
    import faiss
    
    # Load FAISS index
    index = faiss.read_index(faiss_index_path)
    
    # Extract vectors from FAISS index
    d = index.d
    n = index.ntotal
    vectors = np.zeros((n, d), dtype=np.float32)
    for i in range(n):
        vectors[i] = index.reconstruct(i)
    
    # Load metadata
    with open(meta_file_path) as f:
        metadata = json.load(f)
    
    # Upload to Qdrant
    retriever = QdrantRetriever(collection_name=collection_name)
    retriever.upload_vectors(vectors, metadata)
    
    print(f"Uploaded {n} vectors from FAISS index to Qdrant collection {collection_name}")
    return retriever

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        # Upload FAISS index to Qdrant
        faiss_path = sys.argv[1]
        meta_path = sys.argv[2]
        collection = sys.argv[3] if len(sys.argv) > 3 else "multimodal_rag"
        upload_faiss_to_qdrant(faiss_path, meta_path, collection)
    else:
        print("Usage: python qdrant_retriever.py <faiss_index_path> <meta_file_path> [collection_name]")