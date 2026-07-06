from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import os

app = FastAPI()

JD_MODEL_LOCAL_DIR = os.getenv("JD_MODEL_LOCAL_DIR", "/models/jd")
NCS_MODEL_LOCAL_DIR = os.getenv("NCS_MODEL_LOCAL_DIR", "/models/ncs")

jd_tokenizer = AutoTokenizer.from_pretrained(JD_MODEL_LOCAL_DIR)
jd_model = AutoModel.from_pretrained(JD_MODEL_LOCAL_DIR)
ncs_tokenizer = AutoTokenizer.from_pretrained(NCS_MODEL_LOCAL_DIR)
ncs_model = AutoModel.from_pretrained(NCS_MODEL_LOCAL_DIR)

class EmbedRequest(BaseModel):
    text: str

class BatchEmbedRequest(BaseModel):
    texts: list[str]

def encode(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()

@app.post("/embed/jd")
def embed_jd(req: EmbedRequest):
    return {"vector": encode(req.text, jd_tokenizer, jd_model)}

@app.post("/embed/jd/batch")
def embed_jd_batch(req: BatchEmbedRequest):
    return {"vectors": [encode(text, jd_tokenizer, jd_model) for text in req.texts]}

@app.post("/embed/ncs")
def embed_ncs(req: EmbedRequest):
    return {"vector": encode(req.text, ncs_tokenizer, ncs_model)}

@app.post("/embed/ncs/batch")
def embed_ncs_batch(req: BatchEmbedRequest):
    return {"vectors": [encode(text, ncs_tokenizer, ncs_model) for text in req.texts]}

@app.get("/health")
def health():
    return {"status": "ok"}
