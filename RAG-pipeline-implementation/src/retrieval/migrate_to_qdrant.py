# src/retrieval/migrate_to_qdrant.py
import os
import argparse
from pathlib import Path
from qdrant_retriever import upload_faiss_to_qdrant

def migrate_to_qdrant():
    """
    Utility script to migrate existing FAISS index and metadata to a Qdrant collection.
    """
    parser = argparse.ArgumentParser(description="Migrate FAISS index to Qdrant")
    parser.add_argument("--faiss-index", type=str, default="data/index/index.faiss",
                        help="Path to the FAISS index file")
    parser.add_argument("--metadata", type=str, default="data/index/metadata.json",
                        help="Path to the metadata JSON file")
    parser.add_argument("--collection", type=str, default="multimodal_rag",
                        help="Name of the Qdrant collection to create")
    
    args = parser.parse_args()
    
    print(f"Migrating FAISS index from {args.faiss_index} to Qdrant collection {args.collection}...")
    retriever = upload_faiss_to_qdrant(args.faiss_index, args.metadata, args.collection)
    print("Migration complete!")
    
if __name__ == "__main__":
    migrate_to_qdrant()