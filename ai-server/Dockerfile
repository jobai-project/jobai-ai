FROM python:3.11-slim

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY main.py .

# 모델은 Docker 볼륨으로 마운트 (이미지 재빌드 없이 모델 교체 가능)
VOLUME ["/app/models"]

# 포트 8001 노출
EXPOSE 8001

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
