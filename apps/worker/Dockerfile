FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cpu
ENV MODEL_WEIGHTS_PATH=weights/unet3d_final_fp32.pth

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

