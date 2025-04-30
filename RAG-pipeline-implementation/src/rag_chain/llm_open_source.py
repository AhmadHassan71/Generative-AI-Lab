# src/rag_chain/llm_open_source.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def load_local_llm(
    model_name: str = "reasonwang/google-flan-t5-small-alpaca",
    max_new_tokens: int = 256,
):
    """
    Load a small FLAN-T5 model onto CPU (or GPU if available), with no device_map.
    """
    # 1) tokenizer + model (no device_map!)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True),
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # 2) move to GPU if available, else CPU
    if torch.cuda.is_available():
        model = model.to("cuda")
        device = 0
    else:
        model = model.to("cpu")
        device = -1

    # 3) build HF pipeline
    text_pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=model_name,      # ← pass the string name instead
    device=device,
    max_new_tokens=max_new_tokens,
    do_sample=False,
    temperature=0.0,
)

    return text_pipe
