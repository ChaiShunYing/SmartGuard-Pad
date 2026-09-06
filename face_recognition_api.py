"""
SmartGuard Pad - Cloud face recognition API (FastAPI version)
When ESP32 detects a weight drop, it takes a photo and POSTs it to /recognize,
which runs InsightFace comparison and returns the result directly in the response.
"""

import pickle
import numpy as np
import cv2
from fastapi import FastAPI, Request
from insightface.app import FaceAnalysis

# ============ Config ============
THRESHOLD = 0.5

# ============ Initialize insightface (loaded once at startup) ============
app = FastAPI()
face_app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(320, 320))

with open("owner_profile.pkl", "rb") as f:
    owner_embeddings = pickle.load(f)

print(f"Loaded {len(owner_embeddings)} owner embeddings")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def check_is_owner(face_embedding):
    best_score = max(cosine_similarity(face_embedding, e) for e in owner_embeddings)
    return best_score >= THRESHOLD, best_score


@app.post("/recognize")
async def recognize(request: Request):
    # ESP32 sends raw JPEG bytes directly (not multipart/form-data),
    # so read the raw request body instead of using UploadFile
    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"result": "not_owner", "similarity": 0.0, "reason": "invalid_image"}

    faces = face_app.get(frame)

    if len(faces) == 0:
        return {"result": "not_owner", "similarity": 0.0, "reason": "no_face_detected"}

    is_owner, score = check_is_owner(faces[0].embedding)
    result = "owner" if is_owner else "not_owner"
    print(f"Recognition result: {result}, similarity: {score:.4f}")

    return {"result": result, "similarity": round(float(score), 4), "reason": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}
