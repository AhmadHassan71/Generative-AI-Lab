# src/ui/app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from rag_chain.chain import answer
from rag_chain.chain import text_model
from retrieval.qdrant_retriever import QdrantRetriever
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# load retriever & CLIP
retr = QdrantRetriever(
    collection_name=os.getenv("QDRANT_COLLECTION", "multimodal_rag"),
    meta_file="data/index/metadata.json"
)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

st.title("📚 Multimodal RAG Demo with GPT-4o Mini & Qdrant")
query = st.text_input("Enter a text question:")
img = st.file_uploader("…or upload an image", type=["png","jpg"])

if st.button("Submit"):
    if query:
        ans = answer(query)
        st.markdown("**Answer:**")
        st.write(ans)
    elif img:
        with open("temp.png","wb") as f: f.write(img.getbuffer())
        img_pil = Image.open("temp.png").convert("RGB")
        inp = clip_proc(images=img_pil, return_tensors="pt")
        with torch.no_grad():
            iv = clip_model.get_image_features(**inp).cpu().numpy().astype("float32")
        hits = retr.query(iv, top_k=5)
        st.image(img_pil, caption="Your image")
        st.write(answer("Describe this image."))
