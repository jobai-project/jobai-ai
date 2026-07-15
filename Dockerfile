FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY ai-server/requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install awscli

COPY ai-server/ .
COPY entrypoint.sh .

RUN python -c "from sentence_transformers import CrossEncoder; model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'); model.save('/models/rerank')"

RUN chmod +x entrypoint.sh

EXPOSE 8001

ENTRYPOINT ["./entrypoint.sh"]
