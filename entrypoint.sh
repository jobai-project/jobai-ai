#!/bin/sh
set -eu

: "${JD_MODEL_LOCAL_DIR:=/models/jd}"
: "${NCS_MODEL_LOCAL_DIR:=/models/ncs}"
: "${JD_MODEL_S3_PREFIX:=models/jd/}"
: "${NCS_MODEL_S3_PREFIX:=models/ncs/}"

sync_model() {
  model_name="$1"
  s3_prefix="$2"
  local_dir="$3"

  if [ -f "$local_dir/config.json" ] && [ -f "$local_dir/model.safetensors" ]; then
    echo "$model_name model already exists at $local_dir"
    return
  fi

  if [ -z "${AI_MODEL_S3_BUCKET:-}" ]; then
    echo "AI_MODEL_S3_BUCKET is required to download $model_name model" >&2
    exit 1
  fi

  mkdir -p "$local_dir"
  aws s3 sync "s3://$AI_MODEL_S3_BUCKET/$s3_prefix" "$local_dir"
}

sync_model "JD" "$JD_MODEL_S3_PREFIX" "$JD_MODEL_LOCAL_DIR"
sync_model "NCS" "$NCS_MODEL_S3_PREFIX" "$NCS_MODEL_LOCAL_DIR"

exec uvicorn main:app --host 0.0.0.0 --port 8001
