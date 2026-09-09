FROM python:3.11-slim

# Dépendances système nécessaires à OpenCV / onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Hugging Face Spaces (SDK Docker) route le trafic vers le port 7860 par défaut.
ENV DATA_DIR=/data
RUN mkdir -p /data/photos /data/thumbs
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
