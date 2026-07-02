from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import os

app = FastAPI()

MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "/models/current")
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR)
model = AutoModel.from_pretrained(MODEL_LOCAL_DIR)

class EmbedRequest(BaseModel):
    text: str

def encode(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()

@app.post("/embed")
def embed(req: EmbedRequest):
    return {"vector": encode(req.text)}

@app.get("/health")
def health():
    return {"status": "ok"}