# src/ui/app_fixed.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment variables to prevent conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Import streamlit after setting environment variables
import streamlit as st
from rag_chain.chain import answer, describe_image
from rag_chain.chain import text_model, RELEVANCE_THRESHOLD
from retrieval.qdrant_retriever import QdrantRetriever
from PIL import Image
import numpy as np
import torch
from dotenv import load_dotenv
from rag_chain.llm_openai import load_openai_llm

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client for direct image processing
openai_llm = load_openai_llm(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    max_tokens=int(os.getenv("MAX_TOKENS", "512"))
)

# Disable Streamlit's file watcher in the code
from streamlit.web.server.server import Server
Server._watch_for_local_file_changes = lambda _: None

# Function to process image directly with OpenAI (without RAG)
def process_image_directly(image_path, prompt="Describe this image in detail."):
    """
    Process an image directly using OpenAI's GPT-4o Mini without using the RAG database
    
    Args:
        image_path: Path to the image file
        prompt: Prompt to send to the model
    
    Returns:
        The model's response
    """
    try:
        # Use the OpenAI API to process the image directly
        response = openai_llm(prompt, temp=0.2, image_path=image_path)
        return response
    except Exception as e:
        return f"Error processing image directly: {str(e)}"

# Global variables to avoid reloading models
@st.cache_resource
def load_models():
    from transformers import CLIPProcessor, CLIPModel
    
    # load retriever
    retr = QdrantRetriever(
        collection_name=os.getenv("QDRANT_COLLECTION", "multimodal_rag"),
        meta_file="data/index/clean_metadata.json"
    )
    
    # Load CLIP model
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    return retr, clip_proc, clip_model

# Load models
retr, clip_proc, clip_model = load_models()

# App UI
st.title("📚 Multimodal RAG Demo with GPT-4o Mini & Qdrant")
st.write("Ask questions about the documents or upload an image for analysis.")

# Add relevance threshold control in sidebar
st.sidebar.title("Configuration")
relevance_threshold = st.sidebar.slider(
    "Relevance Threshold", 
    min_value=0.05, 
    max_value=0.50, 
    value=float(os.getenv("RELEVANCE_THRESHOLD", "0.25")),
    step=0.05,
    help="Higher values require better matches. Increase this if getting irrelevant responses."
)

# Set the relevance threshold in environment
os.environ["RELEVANCE_THRESHOLD"] = str(relevance_threshold)

# Create tabs for text and image inputs
tab1, tab2, tab3 = st.tabs(["Text Query", "Image Query with RAG", "Direct Image Processing"])

with tab1:
    query = st.text_input("Enter a text question:")
    if st.button("Submit Question", key="text_submit"):
        if query:
            with st.spinner("Generating answer..."):
                try:
                    ans = answer(query)
                    st.markdown("**Answer:**")
                    st.write(ans)
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

with tab2:
    st.write("Upload an image to retrieve relevant content from the FAST University database")
    img_rag = st.file_uploader("Upload an image for RAG analysis", type=["png","jpg","jpeg"], key="rag_image")
    if st.button("Submit Image for RAG Analysis", key="img_rag_submit"):
        if img_rag:
            with st.spinner("Processing image with RAG..."):
                try:
                    # Save the uploaded image temporarily
                    with open("temp.png", "wb") as f:
                        f.write(img_rag.getbuffer())
                    
                    # Process the image
                    img_pil = Image.open("temp.png").convert("RGB")
                    inp = clip_proc(images=img_pil, return_tensors="pt")
                    
                    # Get the image features
                    with torch.no_grad():
                        iv = clip_model.get_image_features(**inp).cpu().numpy().astype("float32")
                    
                    # Search similar images
                    hits = retr.query(iv, top_k=5)
                    
                    # Display the image
                    st.image(img_pil, caption="Your image")
                    
                    # Show the relevance scores for transparency
                    max_score = max([hit.get("score", 0) for hit in hits]) if hits else 0
                    
                    # Determine if the image is relevant to FAST
                    is_relevant = max_score >= relevance_threshold
                    
                    # Display relevance information
                    st.markdown("### Image Processing Information")
                    if is_relevant:
                        st.success(f"✅ This image appears to be related to FAST University content (highest similarity score: {max_score:.3f})")
                    else:
                        st.warning(f"⚠️ This image doesn't appear to be related to FAST University content (highest similarity score: {max_score:.3f})")
                        st.info("The system will still process this image, but the results may not be accurate since it's outside the knowledge domain of this RAG system.")
                    
                    # Generate answer about the image using the new describe_image function
                    st.markdown("**Image Analysis:**")
                    ans = describe_image("Describe this image")
                    st.write(ans)
                    
                    # Show all relevance scores for transparency
                    with st.expander("View all similarity scores"):
                        st.markdown("**Relevance Scores:**")
                        st.write("The scores below show how similar the uploaded image is to images in our database:")
                        for i, hit in enumerate(hits[:5]):
                            score = hit.get("score", 0)
                            status = "✅ Relevant" if score >= relevance_threshold else "❌ Not relevant"
                            st.write(f"Match {i+1}: {score:.3f} - {status}")
                        
                        st.write("**How the system processes images:**")
                        st.write("""
                        1. The CLIP model extracts visual features from your image
                        2. These features are compared against the vector database
                        3. The system retrieves similar images based on their vector similarity (scores closer to 1 indicate higher similarity)
                        4. The GPT-4o Mini model generates a response based on the retrieved content
                        5. If no relevant matches are found (scores below threshold), the system informs you that it can't provide accurate information
                        """)
                    
                    # Only display relevant images from the database
                    relevant_hits = [hit for hit in hits if hit.get("score", 0) >= relevance_threshold]
                    if relevant_hits:
                        st.markdown("**Similar Images in Database:**")
                        for i, hit in enumerate(relevant_hits):
                            if "path" in hit and os.path.exists(hit["path"]):
                                st.image(hit["path"], caption=f"Match {i+1} - Score: {hit['score']:.2f}")
                                if "caption" in hit and hit["caption"]:
                                    st.write(hit["caption"])
                    else:
                        st.info("No relevant matching images found in the database.")
                        
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

with tab3:
    st.write("Upload an image to process directly with GPT-4o Mini (no RAG database needed)")
    img_direct = st.file_uploader("Upload any image for direct analysis", type=["png","jpg","jpeg"], key="direct_image")
    prompt = st.text_input("Custom prompt (optional):", value="Describe this image in detail.", key="direct_prompt")
    
    if st.button("Analyze Image Directly", key="direct_image_submit"):
        if img_direct:
            with st.spinner("Processing image directly with GPT-4o Mini..."):
                try:
                    # Save the uploaded image temporarily
                    img_path = "temp_direct.png"
                    with open(img_path, "wb") as f:
                        f.write(img_direct.getbuffer())
                    
                    # Display the image
                    img_pil = Image.open(img_path).convert("RGB")
                    st.image(img_pil, caption="Your image")
                    
                    # Process directly using OpenAI
                    st.markdown("**Direct Image Analysis:**")
                    response = process_image_directly(img_path, prompt)
                    st.write(response)
                    
                    st.info("This analysis was performed directly using GPT-4o Mini's image processing capabilities, without querying the RAG database. This works for any image, not just those related to FAST University content.")
                    
                except Exception as e:
                    st.error(f"Error processing image directly: {str(e)}")

# Show information about the models being used
st.sidebar.title("Model Information")
st.sidebar.info(f"LLM: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
st.sidebar.info(f"Vector DB: Qdrant ({os.getenv('QDRANT_COLLECTION', 'multimodal_rag')})")
st.sidebar.info("Embedding: SentenceTransformer (all-mpnet-base-v2)")
st.sidebar.info("Image Model: CLIP (openai/clip-vit-base-patch32)")

# Footer
st.markdown("---")
st.caption("Multimodal RAG System - FAST-NUCES Research Project")