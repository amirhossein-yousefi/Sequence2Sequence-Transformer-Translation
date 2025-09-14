import os
from typing import Any, Dict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from sagemaker_huggingface_inference_toolkit import decoder_encoder


def model_fn(model_dir: str):
    """
    Load the fine-tuned Seq2Seq model and tokenizer from `model_dir`.
    Returns an object the server passes to `transform_fn` (we return a dict).
    """
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    pipe = pipeline("translation", model=model, tokenizer=tokenizer, device=device)
    return {"pipe": pipe, "tokenizer": tokenizer}


def transform_fn(model: Dict[str, Any], input_data: bytes, content_type: str, accept_type: str):
    """
    End-to-end transform: decode -> predict -> encode.
    Supports text or JSON inputs, but JSON is preferred.
    """
    # Decode incoming request into python object
    data = decoder_encoder.decode(input_data, content_type or "application/json")

    # Accept either {"inputs": "...", "parameters": {...}} or a raw string/list
    if isinstance(data, dict):
        inputs = data.get("inputs", "")
        params = data.get("parameters", {}) or {}
    else:
        inputs = data
        params = {}

    pipe = model["pipe"]

    # Call the translation pipeline
    outputs = pipe(inputs, **params)

    # Encode response
    return decoder_encoder.encode(outputs, accept_type or "application/json")