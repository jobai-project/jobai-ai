from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
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


# 공통 schema
class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    vector: list[float]


# 배치 임베딩 schema (id 기반)
class BatchEmbedItem(BaseModel):
    id: int
    text: str


class BatchEmbedRequest(BaseModel):
    items: list[BatchEmbedItem]


class BatchEmbedResponseItem(BaseModel):
    id: int
    vector: list[float]


class BatchEmbedResponse(BaseModel):
    items: list[BatchEmbedResponseItem]


class ResumeEmbedRequest(BaseModel):
    text: str
    model_type: Literal["private", "public"]


class ResumeBatchEmbedRequest(BaseModel):
    items: list[BatchEmbedItem]
    model_type: Literal["private", "public"]


# 사기업 score schema
class ScorePrivateRequest(BaseModel):
    job_id: Optional[str] = None
    title: Optional[str] = None
    job_category: Optional[str] = None

    jd_text: str
    jd_vec: list[float]
    resume_vec: list[float]

    resume_skills: list[str]
    experience_years: int
    resume_role: Optional[str] = None


class ScorePrivateResponse(BaseModel):
    score: float
    above_threshold: bool

    ts: float
    cs: float
    qs: float
    gp: float

    matched_skills: list[str]
    missing_skills: list[str]
    career_met: bool

    score_reason: str
    penalties: list[str]


# 공기업 score schema
class ScorePublicRequest(BaseModel):
    job_id: Optional[str] = None
    title: Optional[str] = None
    company_name: Optional[str] = None
    job_role: Optional[str] = None
    apply_qualification: Optional[str] = None
    html_content: Optional[str] = None

    jd_text: str
    resume: dict
    jd_vec: list[float]
    resume_vec: list[float]


class ScorePublicResponse(BaseModel):
    score: float
    above_threshold: bool

    kw: float
    cs: float
    ws: float
    gp: float

    matched_skills: list[str]
    missing_skills: list[str]
    matched_certs: list[str]
    missing_certs: list[str]

    job_cluster: str
    resume_cluster: str

    score_reason: str
    penalties: list[str]


# embedding 함수
def encode_jd(text: str) -> list[float]:
    inputs = jd_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = jd_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()


def encode_ncs(text: str) -> list[float]:
    return ncs_model.encode(text).tolist()


# 사기업 엔드포인트
@app.post("/embed/jd", response_model=EmbedResponse)
def embed_jd(req: EmbedRequest):
    return {"vector": encode_jd(req.text)}


@app.post("/embed/jd/batch", response_model=BatchEmbedResponse)
def embed_jd_batch(req: BatchEmbedRequest):
    return {
        "items": [
            {"id": item.id, "vector": encode_jd(item.text)}
            for item in req.items
        ]
    }


@app.post("/score/private", response_model=ScorePrivateResponse)
def score_private_endpoint(req: ScorePrivateRequest):
    return score_private(
        jd_text=req.jd_text,
        jd_vec=req.jd_vec,
        resume_vec=req.resume_vec,
        resume_skills=req.resume_skills,
        experience_years=req.experience_years,
        job_category=req.job_category,
        title=req.title,
        resume_role=req.resume_role,
    )


# 공기업 엔드포인트
@app.post("/embed/ncs", response_model=EmbedResponse)
def embed_ncs(req: EmbedRequest):
    return {"vector": encode_ncs(req.text)}


@app.post("/embed/ncs/batch", response_model=BatchEmbedResponse)
def embed_ncs_batch(req: BatchEmbedRequest):
    return {
        "items": [
            {"id": item.id, "vector": encode_ncs(item.text)}
            for item in req.items
        ]
    }


@app.post("/score/public", response_model=ScorePublicResponse)
def score_public_endpoint(req: ScorePublicRequest):
    return score_ncs(
        jd_text=req.jd_text,
        resume=req.resume,
        jd_vec=req.jd_vec,
        resume_vec=req.resume_vec,
        title=req.title,
        company_name=req.company_name,
        job_role=req.job_role,
        apply_qualification=req.apply_qualification,
        html_content=req.html_content,
    )


# 이력서 엔드포인트
@app.post("/embed/resume", response_model=EmbedResponse)
def embed_resume(req: ResumeEmbedRequest):
    if req.model_type == "private":
        return {"vector": encode_jd(req.text)}
    if req.model_type == "public":
        return {"vector": encode_ncs(req.text)}
    raise HTTPException(
        status_code=400,
        detail="model_type must be 'private' or 'public'",
    )


@app.post("/embed/resume/batch", response_model=BatchEmbedResponse)
def embed_resume_batch(req: ResumeBatchEmbedRequest):
    if req.model_type == "private":
        return {
            "items": [
                {"id": item.id, "vector": encode_jd(item.text)}
                for item in req.items
            ]
        }
    if req.model_type == "public":
        return {
            "items": [
                {"id": item.id, "vector": encode_ncs(item.text)}
                for item in req.items
            ]
        }
    raise HTTPException(
        status_code=400,
        detail="model_type must be 'private' or 'public'",
    )


# 헬스체크
@app.get("/health")
def health():
    return {"status": "ok"}