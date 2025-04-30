# src/preprocessing/extract_images.py
import fitz  # PyMuPDF
import json
import sys
from pathlib import Path
import os

def extract_images(pdf_path, out_dir, out_json):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    images = []
    for p in range(len(doc)):
        for img_idx, img in enumerate(doc[p].get_images(full=True), 1):
            try:
                xref = img[0]
                base_pix = fitz.Pixmap(doc, xref)
                
                # Always convert to RGB (safest approach)
                if base_pix.n >= 4:  # CMYK or other colorspaces with alpha
                    pix = fitz.Pixmap(fitz.csRGB, base_pix)
                    base_pix = None  # Free the original pixmap
                else:
                    pix = base_pix
                
                img_path = Path(out_dir) / f"page{p+1}_img{img_idx}.png"
                
                # Use JPEG for problematic images as a fallback
                try:
                    pix.save(str(img_path))
                except ValueError:
                    # If PNG fails, try saving as JPEG
                    img_path = Path(out_dir) / f"page{p+1}_img{img_idx}.jpg"
                    pix.save(str(img_path), output="jpeg")
                    
                images.append({"page": p+1, "path": str(img_path)})
                
            except Exception as e:
                print(f"Warning: Could not extract image {img_idx} from page {p+1}: {e}")
                continue
            
    with open(out_json, "w") as f:
        json.dump(images, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Error: Missing required arguments.")
        print("Usage: python extract_images.py <pdf_path> <output_image_directory> <output_json_path>")
        print("Example: python extract_images.py data/raw_pdfs/1.pdf data/images/1 data/images/1.json")
        sys.exit(1)
    
    pdf, img_dir, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    extract_images(pdf, img_dir, out_json)
    print(f"Extracted images metadata to {out_json}")
