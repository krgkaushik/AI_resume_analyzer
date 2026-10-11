import pickle
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="AI Resume ATS Analyzer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

try:
    with open(BASE_DIR / "resume_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(BASE_DIR / "resume_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
except Exception as exc:
    raise RuntimeError(f"Unable to load saved model files: {exc}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    cleaned = clean_text(text)
    return re.findall(r"\b[a-z]{3,}\b", cleaned)


def compute_ats_score(resume_text: str, jd_text: str):
    resume_tokens = set(tokenize(resume_text))
    jd_tokens = set(tokenize(jd_text))

    if not jd_tokens:
        return {
            "ats_score": 0.0,
            "match_percentage": 0.0,
            "missing_keywords": [],
            "matched_keywords": [],
        }

    matched = sorted(jd_tokens & resume_tokens)
    missing = sorted(jd_tokens - resume_tokens)
    coverage = (len(matched) / len(jd_tokens)) * 100 if jd_tokens else 0

    try:
        tfidf_matrix = vectorizer.transform([resume_text, jd_text])
        similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        similarity_pct = max(0.0, min(1.0, float(similarity))) * 100
    except Exception:
        similarity_pct = 0.0

    try:
        resume_vector = vectorizer.transform([resume_text])
        probabilities = classifier.predict_proba(resume_vector)[0]
        model_confidence = float(np.max(probabilities)) * 100
        predicted_label = classifier.classes_[int(np.argmax(probabilities))]
    except Exception:
        model_confidence = 0.0
        predicted_label = "Unknown"

    ats_score = min(100.0, round(0.55 * coverage + 0.30 * similarity_pct + 0.15 * model_confidence, 2))
    match_percentage = round(min(100.0, max(0.0, coverage + (similarity_pct * 0.35))), 2)

    return {
        "ats_score": ats_score,
        "match_percentage": match_percentage,
        "missing_keywords": missing[:10],
        "matched_keywords": matched[:10],
        "predicted_industry": predicted_label,
        "model_confidence": round(model_confidence, 2),
    }


@app.get("/")
def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/score")
def score_resume(payload: dict):
    resume_text = (payload.get("resumeText") or "").strip()
    jd_text = (payload.get("jdText") or "").strip()

    if not resume_text or not jd_text:
        raise HTTPException(status_code=400, detail="Both resume text and job description are required.")

    result = compute_ats_score(resume_text, jd_text)
    return result


@app.get("/health")
def health_check():
    return {"status": "ok"}
