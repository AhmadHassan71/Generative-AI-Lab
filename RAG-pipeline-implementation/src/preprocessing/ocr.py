# src/preprocessing/ocr.py
import pytesseract
from PIL import Image
import json

def ocr_images(metadata_json, out_json):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    with open(metadata_json) as f:
        images = json.load(f)
    for img in images:
        text = pytesseract.image_to_string(Image.open(img["path"]))
        img["caption"] = text
    with open(out_json, "w") as f:
        json.dump(images, f, indent=2)

if __name__ == "__main__":
    import sys
    ocr_images(sys.argv[1], sys.argv[2])
    print(f"OCRed images, wrote to {sys.argv[2]}")
