# src/preprocessing/extract_text.py
import fitz  # PyMuPDF
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
        r'FAST-NUCES [0-9]+ ',
        r'\d+/\d+',
        r'RN Stationers.*',
        r'BS FINAL YEAR PROJECT HANDBOOK \d+.*',
        r'^National University.*\[Campus\]$',
        r'^\s*\d+\s*$'  # Page numbers alone on a line
    ]
    
    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Remove multiple spaces and trim
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def extract_text(pdf_path, out_json):
    pages = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, 1):
        # Extract text
        raw_text = page.get_text() or ""
        
        # Clean the text to remove headers and footers
        cleaned_text = clean_text(raw_text)
        
        # Add to pages if there's meaningful content left after cleaning
        if len(cleaned_text.strip()) > 20:  # Only include if there's meaningful content
            pages.append({"page": i, "text": cleaned_text})
    
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=2)

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1]        # e.g. data/raw_pdfs/1.pdf
    out = sys.argv[2]        # e.g. data/text_pages/1.json
    extract_text(pdf, out)
    print(f"Extracted clean text to {out}")
