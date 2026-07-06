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

RUN chmod +x entrypoint.sh

EXPOSE 8001

ENTRYPOINT ["./entrypoint.sh"]
