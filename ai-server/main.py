from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# 파인튜닝된 모델 로드 (서버 시작 시 한 번만 실행)
# 실데이터 파인튜닝 전까지는 베이스 모델 사용
model = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")

# 단일 요청 형식
class EmbedRequest(BaseModel):
    text: str

# 단일 응답 형식
class EmbedResponse(BaseModel):
    vector: list[float]

# 배치 요청 형식
class EmbedBatchRequest(BaseModel):
    texts: list[str]

# 배치 응답 형식
class EmbedBatchResponse(BaseModel):
    vectors: list[list[float]]

@app.post("/embed")
def embed(req: EmbedRequest) -> EmbedResponse:
    # 텍스트를 768차원 벡터로 변환
    vector = model.encode(req.text).tolist()
    return EmbedResponse(vector=vector)

@app.post("/embed/batch")
def embed_batch(req: EmbedBatchRequest) -> EmbedBatchResponse:
    # 여러 텍스트를 한 번에 임베딩
    vectors = model.encode(req.texts).tolist()
    return EmbedBatchResponse(vectors=vectors)

@app.get("/health")
def health():
    # Docker 헬스체크용
    return {"status": "ok"}
