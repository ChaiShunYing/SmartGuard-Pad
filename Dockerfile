FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_m', providers=['CPUExecutionProvider'])" || true

RUN if [ -d "/root/.insightface/models/buffalo_m/buffalo_m" ]; then \
        mv /root/.insightface/models/buffalo_m/buffalo_m/* /root/.insightface/models/buffalo_m/ && \
        rmdir /root/.insightface/models/buffalo_m/buffalo_m; \
    fi

RUN python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_m', providers=['CPUExecutionProvider']).prepare(ctx_id=0, det_size=(320,320))"

COPY . .
EXPOSE 10000
CMD ["uvicorn", "face_recognition_api:app", "--host", "0.0.0.0", "--port", "10000"]
