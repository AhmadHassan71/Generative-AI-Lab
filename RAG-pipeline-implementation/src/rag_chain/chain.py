# src/rag_chain/chain.py

import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import the components using correct relative paths
from src.retrieval.qdrant_retriever import QdrantRetriever
from src.rag_chain.llm_openai import load_openai_llm

# Load environment variables
load_dotenv()

# 1) Embedder & Retriever
text_model = SentenceTransformer("all-mpnet-base-v2")

# Initialize QdrantRetriever with metadata for backward compatibility
retriever = QdrantRetriever(
    collection_name=os.getenv("QDRANT_COLLECTION", "multimodal_rag"),
    meta_file="data/index/metadata.json"
)

# 2) LLM pipeline
llm = load_openai_llm(
    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    max_tokens=int(os.getenv("MAX_TOKENS", "256"))
)

# Similarity threshold for determining relevance
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))

def answer(query: str, top_k: int = 5, image_query: bool = False) -> str:
    """
    RAG flow with relevance check: embed → retrieve → check relevance → summarise.
    
    Args:
        query: The query text
        top_k: Number of results to retrieve
        image_query: Whether this is for an image query (affects response)
        
    Returns:
        Response from the LLM
    """
    # Embed the query
    qv = text_model.encode([query], normalize_embeddings=True).astype("float32")
    
    # Retrieve results
    hits = retriever.query(qv, top_k)
    
    # Check if the results are relevant by examining similarity scores
    if not hits or max(hit.get("score", 0) for hit in hits) < RELEVANCE_THRESHOLD:
        if image_query:
            return "I don't have specific information about this image. It doesn't appear to be related to the documents in my knowledge base. Please try with an image that relates to the university documents I've been trained on."
        else:
            return "I don't have enough relevant information to answer this question. Please ask something related to the university documents in my knowledge base."
    
    # Build context from relevant results
    context = "\n\n".join(c.get("text", c.get("caption","")).strip() for c in hits)
    
    # Add relevance scores to help the model judge confidence
    scores_info = "\n".join([f"Result {i+1} relevance: {hit.get('score', 0):.2f}" 
                            for i, hit in enumerate(hits)])
    
    # Prompt with instructions to check relevance
    prompt = (
        "You are a helpful assistant specialized in answering questions about university documents. "
        "You have been provided with context information retrieved from a vector database. "
        "Read the context below and the relevance scores, then respond to the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Relevance scores (higher is better, scores below 0.25 may indicate irrelevant results):\n{scores_info}\n\n"
        f"Question: {query}\n\n"
        "Important instructions:\n"
        "1. If the context doesn't contain relevant information to the question, state that you don't have enough information to answer.\n"
        "2. If the question is completely unrelated to university documents, politely explain that you're specialized in answering questions about university content.\n"
        "3. If you have relevant information, answer in 2-3 concise sentences in your own words.\n"
        "4. Do not include page references in your answer.\n"
        "5. Do not refer to 'the context' or 'the document' in your answer.\n\n"
        "Answer:"
    )
    
    # Call LLM
    return llm(prompt)

def describe_image(image_description: str = "Describe this image") -> str:
    """
    Handle image description requests with specific instructions to check relevance.
    
    Args:
        image_description: The query to describe the image
        
    Returns:
        Response from the LLM
    """
    return answer(image_description, top_k=5, image_query=True)
