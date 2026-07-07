from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from scoring_jd import score_private
from scoring_ncs import score_ncs
import torch
import os

app = FastAPI()

# 사기업 모델
JD_MODEL_DIR = os.getenv("JD_MODEL_LOCAL_DIR", "/models/jd")
jd_tokenizer = AutoTokenizer.from_pretrained(JD_MODEL_DIR)
jd_model = AutoModel.from_pretrained(JD_MODEL_DIR)

# 공기업 모델
NCS_MODEL_DIR = os.getenv("NCS_MODEL_LOCAL_DIR", "/models/ncs")
ncs_model = SentenceTransformer(NCS_MODEL_DIR)


class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    vector: list[float]

class BatchEmbedRequest(BaseModel):
    texts: list[str]

class BatchEmbedResponse(BaseModel):
    vectors: list[list[float]]

class ScorePrivateRequest(BaseModel):
    jd_text: str
    jd_vec: list[float]
    resume_vec: list[float]
    resume_skills: list[str]
    experience_years: int

class ScorePublicRequest(BaseModel):
    jd_text: str
    resume: dict
    jd_vec: list[float]
    resume_vec: list[float]

class ScorePrivateResponse(BaseModel):
    score: float
    above_threshold: bool
    matched_skills: list[str]
    missing_skills: list[str]
    career_met: bool
    score_reason: str
    penalties: list[str]
    model_version: str

class ScorePublicResponse(BaseModel):
    score: float
    above_threshold: bool
    Kw: float
    Cs: float
    Ws: float
    matched_skills: list[str]
    missing_skills: list[str]
    missing_certs: list[str]
    job_cluster: str
    resume_cluster: str
    score_reason: str
    penalties: list[str]
    model_version: str


def encode_jd(text: str) -> list[float]:
    inputs = jd_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = jd_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()


# 사기업 엔드포인트
@app.post("/embed/jd", response_model=EmbedResponse)
def embed_jd(req: EmbedRequest):
    return {"vector": encode_jd(req.text)}

@app.post("/embed/jd/batch", response_model=BatchEmbedResponse)
def embed_jd_batch(req: BatchEmbedRequest):
    return {"vectors": [encode_jd(text) for text in req.texts]}

@app.post("/score/private", response_model=ScorePrivateResponse)
def score_private_endpoint(req: ScorePrivateRequest):
    return score_private(
        jd_text=req.jd_text,
        jd_vec=req.jd_vec,
        resume_vec=req.resume_vec,
        resume_skills=req.resume_skills,
        experience_years=req.experience_years,
    )


# 공기업 엔드포인트
@app.post("/embed/ncs", response_model=EmbedResponse)
def embed_ncs(req: EmbedRequest):
    return {"vector": ncs_model.encode(req.text).tolist()}

@app.post("/embed/ncs/batch", response_model=BatchEmbedResponse)
def embed_ncs_batch(req: BatchEmbedRequest):
    return {"vectors": ncs_model.encode(req.texts).tolist()}

@app.post("/score/public", response_model=ScorePublicResponse)
def score_public_endpoint(req: ScorePublicRequest):
    return score_ncs(
        jd_text=req.jd_text,
        resume=req.resume,
        jd_vec=req.jd_vec,
        resume_vec=req.resume_vec,
    )


# 이력서 엔드포인트
@app.post("/embed/resume", response_model=EmbedResponse)
def embed_resume(req: EmbedRequest):
    # 사기업 모델 사용 (이력서 전용 모델 없음)
    return {"vector": encode_jd(req.text)}

@app.post("/embed/resume/batch", response_model=BatchEmbedResponse)
def embed_resume_batch(req: BatchEmbedRequest):
    # 사기업 모델 사용 (이력서 전용 모델 없음)
    return {"vectors": [encode_jd(text) for text in req.texts]}


# 헬스체크
@app.get("/health")
def health():
    return {"status": "ok"}