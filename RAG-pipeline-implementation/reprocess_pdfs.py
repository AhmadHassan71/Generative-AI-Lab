#!/usr/bin/env python
# reprocess_pdfs.py

import os
import sys
import glob
from pathlib import Path
import subprocess
import json
from src.preprocessing.extract_text import extract_text
from src.chunking.chunker import chunk_text

def reprocess_pdfs(pdf_dir, text_pages_dir, chunks_dir):
    """
    Reprocess PDF files with the improved text extraction
    that filters out headers and footers.
    
    Args:
        pdf_dir: Directory containing PDF files
        text_pages_dir: Directory to store extracted text pages
        chunks_dir: Directory to store text chunks
    """
    # Create output directories if they don't exist
    Path(text_pages_dir).mkdir(parents=True, exist_ok=True)
    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        pdf_id = os.path.splitext(pdf_name)[0]
        
        text_json = os.path.join(text_pages_dir, f"{pdf_id}.json")
        chunks_json = os.path.join(chunks_dir, f"{pdf_id}.json")
        
        print(f"\nProcessing {pdf_name}...")
        
        # Extract clean text
        print(f"Extracting clean text from {pdf_name}...")
        try:
            extract_text(pdf_path, text_json)
            print(f"Successfully extracted text to {text_json}")
        except Exception as e:
            print(f"Error extracting text from {pdf_name}: {str(e)}")
            continue
            
        # Create chunks
        print(f"Creating chunks from {text_json}...")
        try:
            chunk_text(text_json, chunks_json)
            print(f"Successfully created chunks in {chunks_json}")
        except Exception as e:
            print(f"Error creating chunks from {text_json}: {str(e)}")
            continue
            
        # Analyze the chunks
        try:
            with open(chunks_json, 'r') as f:
                chunks = json.load(f)
            print(f"Created {len(chunks)} chunks from {pdf_name}")
            
            # Check for potential header/footer patterns
            header_footer_patterns = [
                "FAST-NUCES", 
                "BS Final Year Project Handbook", 
                "Page", 
                "National University"
            ]
            
            pattern_found = False
            for pattern in header_footer_patterns:
                for chunk in chunks[:10]:  # Check first 10 chunks
                    if pattern in chunk.get("text", ""):
                        print(f"Warning: Pattern '{pattern}' found in chunks. May need additional filtering.")
                        pattern_found = True
                        break
                if pattern_found:
                    break
                    
        except Exception as e:
            print(f"Error analyzing chunks: {str(e)}")
    
    print("\nProcessing complete!")
    print("\nNext steps:")
    print("1. Verify the quality of the extracted text and chunks")
    print("2. Generate embeddings for the chunks")
    print("3. Build a new index with the clean embeddings")

if __name__ == "__main__":
    # Default directories
    pdf_dir = "data/raw_pdfs"
    text_pages_dir = "data/clean_text_pages"
    chunks_dir = "data/clean_chunks"
    
    # Process command line arguments if provided
    if len(sys.argv) > 1:
        pdf_dir = sys.argv[1]
    if len(sys.argv) > 2:
        text_pages_dir = sys.argv[2]
    if len(sys.argv) > 3:
        chunks_dir = sys.argv[3]
    
    print(f"Processing PDFs from: {pdf_dir}")
    print(f"Saving extracted text to: {text_pages_dir}")
    print(f"Saving chunks to: {chunks_dir}")
    
    reprocess_pdfs(pdf_dir, text_pages_dir, chunks_dir)