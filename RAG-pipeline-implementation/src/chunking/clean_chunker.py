# src/chunking/clean_chunker.py
import json
import re
from pathlib import Path

def clean_text(text):
    """
    Clean text by removing headers, footers, and page numbers.
    """
    # Remove common header/footer patterns
    patterns = [
        r'Page \d+ of \d+',
        r'FAST-NUCES \d+ BS Final Year Project Handbook \d+',
        r'\d+/\d+',
        r'RN Stationers.*',
        r'BS FINAL YEAR PROJECT HANDBOOK \d+.*',
        r'^National University.*\[Campus\]$'
    ]
    
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Remove multiple spaces and trim
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def chunk_text_clean(pages_json, out_json, min_chunk_len=50, max_words=300, overlap=50):
    """
    Create cleaned text chunks from pages.
    
    Args:
        pages_json: Path to JSON file containing page texts
        out_json: Path to output JSON file for chunks
        min_chunk_len: Minimum length of a chunk in characters (to avoid empty chunks)
        max_words: Maximum number of words per chunk
        overlap: Number of words to overlap between chunks
    """
    with open(pages_json) as f:
        pages = json.load(f)
    
    chunks = []
    
    for pg in pages:
        page_num = pg.get("page", 0)
        if isinstance(page_num, str) and page_num.isdigit():
            page_num = int(page_num)
        
        # Clean the text
        text = pg.get("text", "")
        cleaned_text = clean_text(text)
        
        # Skip if chunk is too small after cleaning
        if len(cleaned_text) < min_chunk_len:
            continue
            
        # Split into sentences and create meaningful chunks
        sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
        
        # Initialize the first chunk
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence exceeds the max words, save the chunk and start a new one
            if current_word_count + sentence_word_count > max_words and current_word_count > 0:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= min_chunk_len:
                    chunks.append({"page": page_num, "text": chunk_text})
                
                # Start a new chunk with overlap
                overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_words + sentence_words
                current_word_count = len(current_chunk)
            else:
                # Add the sentence to the current chunk
                current_chunk.extend(sentence_words)
                current_word_count += sentence_word_count
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= min_chunk_len:
                chunks.append({"page": page_num, "text": chunk_text})
    
    # Create output directory if it doesn't exist
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    
    # Write chunks to output file
    with open(out_json, "w") as f:
        json.dump(chunks, f, indent=2)
    
    print(f"Created {len(chunks)} clean text chunks in {out_json}")
    return chunks

def merge_small_chunks(chunks, min_chunk_len=100):
    """
    Merge small chunks with adjacent chunks from the same page to avoid tiny chunks.
    
    Args:
        chunks: List of chunk dictionaries
        min_chunk_len: Minimum length of text in characters
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    # Sort by page number
    sorted_chunks = sorted(chunks, key=lambda x: x["page"])
    
    merged_chunks = []
    current_chunk = None
    
    for chunk in sorted_chunks:
        if not current_chunk:
            current_chunk = chunk.copy()
            continue
            
        # If chunks are from the same page and current is too small
        if chunk["page"] == current_chunk["page"] and len(current_chunk["text"]) < min_chunk_len:
            # Merge with current chunk
            current_chunk["text"] += " " + chunk["text"]
        else:
            # Save the current chunk and start a new one
            if len(current_chunk["text"]) >= min_chunk_len:
                merged_chunks.append(current_chunk)
            elif merged_chunks:
                # If current chunk is too small, append to the previous chunk
                merged_chunks[-1]["text"] += " " + current_chunk["text"]
            else:
                # If it's the first chunk and it's too small, keep it anyway
                merged_chunks.append(current_chunk)
                
            current_chunk = chunk.copy()
    
    # Don't forget the last chunk
    if current_chunk:
        if len(current_chunk["text"]) >= min_chunk_len:
            merged_chunks.append(current_chunk)
        elif merged_chunks:
            merged_chunks[-1]["text"] += " " + current_chunk["text"]
        else:
            merged_chunks.append(current_chunk)
    
    return merged_chunks

def process_text_pages(pages_json, out_json, min_chunk_len=50, max_words=300, overlap=50):
    """
    Process text pages to create clean, meaningful chunks.
    
    Args:
        pages_json: Path to JSON file containing page texts
        out_json: Path to output JSON file for chunks
        min_chunk_len: Minimum length of a chunk in characters
        max_words: Maximum number of words per chunk
        overlap: Number of words to overlap between chunks
    """
    # Create initial chunks
    chunks = chunk_text_clean(pages_json, out_json, min_chunk_len, max_words, overlap)
    
    # Merge small chunks
    merged_chunks = merge_small_chunks(chunks, min_chunk_len * 2)
    
    # Write merged chunks to output file
    with open(out_json, "w") as f:
        json.dump(merged_chunks, f, indent=2)
    
    print(f"Processed into {len(merged_chunks)} clean, merged text chunks in {out_json}")
    return merged_chunks

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python clean_chunker.py <input_pages_json> <output_chunks_json>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_text_pages(input_file, output_file)
    print("Text chunking with header/footer cleaning completed.")