#!/usr/bin/env python
# clean_and_rechunk.py

import os
import sys
import glob
import json
from pathlib import Path
from src.chunking.clean_chunker import process_text_pages

def process_all_text_files(text_pages_dir, output_chunks_dir):
    """
    Process all text page files in the given directory and generate cleaned chunks.
    
    Args:
        text_pages_dir: Directory containing text page JSON files
        output_chunks_dir: Directory to store cleaned chunk JSON files
    """
    # Create output directory if it doesn't exist
    Path(output_chunks_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all JSON files in the text pages directory
    text_files = glob.glob(os.path.join(text_pages_dir, "*.json"))
    
    if not text_files:
        print(f"No JSON files found in {text_pages_dir}")
        return
    
    print(f"Found {len(text_files)} text files to process")
    
    # Process each file
    for file_path in text_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_chunks_dir, f"clean_{file_name}")
        
        print(f"Processing {file_name}...")
        try:
            process_text_pages(file_path, output_path)
            print(f"Successfully processed {file_name} -> {output_path}")
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
    
    print("Text cleaning and chunking complete!")

def compare_chunk_counts(original_chunks_dir, clean_chunks_dir):
    """
    Compare the number of chunks in original and clean directories.
    
    Args:
        original_chunks_dir: Directory containing original chunk JSON files
        clean_chunks_dir: Directory containing cleaned chunk JSON files
    """
    orig_files = glob.glob(os.path.join(original_chunks_dir, "*.json"))
    clean_files = glob.glob(os.path.join(clean_chunks_dir, "*.json"))
    
    print("\n=== Comparison of Original vs Cleaned Chunks ===")
    
    for orig_path in orig_files:
        file_name = os.path.basename(orig_path)
        clean_path = os.path.join(clean_chunks_dir, f"clean_{file_name}")
        
        if os.path.exists(clean_path):
            try:
                with open(orig_path, 'r') as f:
                    orig_chunks = json.load(f)
                
                with open(clean_path, 'r') as f:
                    clean_chunks = json.load(f)
                
                orig_count = len(orig_chunks)
                clean_count = len(clean_chunks)
                
                print(f"{file_name}: Original: {orig_count} chunks, Cleaned: {clean_count} chunks")
                
                # Check for empty or very short chunks
                orig_short = sum(1 for c in orig_chunks if len(c.get("text", "")) < 50)
                clean_short = sum(1 for c in clean_chunks if len(c.get("text", "")) < 50)
                
                if orig_short > 0 or clean_short > 0:
                    print(f"  - Short chunks (<50 chars): Original: {orig_short}, Cleaned: {clean_short}")
                
            except Exception as e:
                print(f"Error comparing {file_name}: {str(e)}")
        else:
            print(f"{file_name}: No corresponding cleaned file found")

if __name__ == "__main__":
    # Default directories
    text_pages_dir = "data/text_pages"
    original_chunks_dir = "data/chunks"
    clean_chunks_dir = "data/clean_chunks"
    
    # Process command line arguments if provided
    if len(sys.argv) > 1:
        text_pages_dir = sys.argv[1]
    if len(sys.argv) > 2:
        clean_chunks_dir = sys.argv[2]
    
    print(f"Processing text pages from: {text_pages_dir}")
    print(f"Saving clean chunks to: {clean_chunks_dir}")
    
    # Process all text files
    process_all_text_files(text_pages_dir, clean_chunks_dir)
    
    # Compare original chunks with clean chunks
    compare_chunk_counts(original_chunks_dir, clean_chunks_dir)
    
    print("\nNext steps:")
    print("1. Verify the quality of the clean chunks")
    print("2. Generate embeddings using your text_embedder.py:")
    print(f"   python src/embeddings/text_embedder.py {clean_chunks_dir}/clean_text_chunks.json data/embeddings/clean_text.npz")
    print("3. Build a new index with the clean embeddings")
    print("4. Update your RAG chain to use the new clean index")