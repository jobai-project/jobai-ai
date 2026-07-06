from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import os

app = FastAPI()

JD_MODEL_DIR = os.getenv("JD_MODEL_LOCAL_DIR", "/models/jd")
NCS_MODEL_DIR = os.getenv("NCS_MODEL_LOCAL_DIR", "/models/ncs")

jd_tokenizer = AutoTokenizer.from_pretrained(JD_MODEL_DIR)
jd_model = AutoModel.from_pretrained(JD_MODEL_DIR)
ncs_model = SentenceTransformer(NCS_MODEL_DIR)


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    vector: list[float]


class BatchEmbedRequest(BaseModel):
    texts: list[str]


class BatchEmbedResponse(BaseModel):
    vectors: list[list[float]]


def encode_jd(text: str) -> list[float]:
    inputs = jd_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = jd_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()


@app.post("/embed/jd", response_model=EmbedResponse)
def embed_jd(req: EmbedRequest):
    return {"vector": encode_jd(req.text)}


@app.post("/embed/jd/batch", response_model=BatchEmbedResponse)
def embed_jd_batch(req: BatchEmbedRequest):
    return {"vectors": [encode_jd(text) for text in req.texts]}


@app.post("/embed/ncs", response_model=EmbedResponse)
def embed_ncs(req: EmbedRequest):
    return {"vector": ncs_model.encode(req.text).tolist()}


@app.post("/embed/ncs/batch", response_model=BatchEmbedResponse)
def embed_ncs_batch(req: BatchEmbedRequest):
    return {"vectors": ncs_model.encode(req.texts).tolist()}


@app.get("/health")
def health():
    return {"status": "ok"}
