"""
SmartGuard Pad - Cloud face recognition API (FastAPI version)
When ESP32 detects a weight drop, it takes a photo and POSTs it to /recognize,
which runs InsightFace comparison, logs the event to Firestore + Storage,
and returns the result directly in the response.
"""
import os
import json
import uuid
import datetime
import pickle
import numpy as np
import cv2
from fastapi import FastAPI, Request
from insightface.app import FaceAnalysis
import firebase_admin
from firebase_admin import credentials, firestore, storage

# ============ Config ============
THRESHOLD = 0.5

# ============ Initialize Firebase Admin (loaded once at startup) ============
firebase_creds = json.loads(os.environ["FIREBASE_CREDS_JSON"])
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred, {
    "storageBucket": "smartguard-pad-system.firebasestorage.app"
})
db = firestore.client()
bucket = storage.bucket()

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


def log_event(device_id: str, image_bytes: bytes, result: str, similarity: float, reason: str):
    """Upload the photo to Storage and write an event record to Firestore."""
    event_id = str(uuid.uuid4())
    try:
        blob = bucket.blob(f"events/{device_id}/{event_id}.jpg")
        blob.upload_from_string(image_bytes, content_type="image/jpeg")
        blob.make_public()
        photo_url = blob.public_url
    except Exception as e:
        print(f"Storage upload failed: {e}")
        photo_url = None

    try:
        event_ref = (
            db.collection("devices").document(device_id)
              .collection("events").document(event_id)
        )
        event_ref.set({
            "timestamp": datetime.datetime.utcnow(),
            "result": result,
            "similarity": similarity,
            "reason": reason,
            "photo_url": photo_url,
        })
    except Exception as e:
        print(f"Firestore write failed: {e}")

    return event_id


@app.post("/recognize")
async def recognize(request: Request):
    device_id = request.query_params.get("device_id", "unknown-device")

    # ESP32 sends raw JPEG bytes directly (not multipart/form-data),
    # so read the raw request body instead of using UploadFile
    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        print("Recognition result: not_owner, reason: invalid_image")
        event_id = log_event(device_id, contents, "not_owner", 0.0, "invalid_image")
        return {"result": "not_owner", "similarity": 0.0, "reason": "invalid_image", "event_id": event_id}

    faces = face_app.get(frame)
    if len(faces) == 0:
        print("Recognition result: not_owner, reason: no_face_detected")
        event_id = log_event(device_id, contents, "not_owner", 0.0, "no_face_detected")
        return {"result": "not_owner", "similarity": 0.0, "reason": "no_face_detected", "event_id": event_id}

    is_owner, score = check_is_owner(faces[0].embedding)
    result = "owner" if is_owner else "not_owner"
    print(f"Recognition result: {result}, similarity: {score:.4f}")

    event_id = log_event(device_id, contents, result, round(float(score), 4), "ok")
    return {"result": result, "similarity": round(float(score), 4), "reason": "ok", "event_id": event_id}


@app.get("/health")
async def health():
    return {"status": "ok"}
