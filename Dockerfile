FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码（.dockerignore 已排除 .venv/.idea 等）
COPY app ./app
COPY db ./db
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.api.production:app", "--host", "0.0.0.0", "--port", "8000"]
