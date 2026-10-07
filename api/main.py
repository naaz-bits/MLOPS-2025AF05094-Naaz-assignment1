"""FastAPI service for heart-disease risk prediction.

Endpoints:
    GET  /health   liveness/readiness probe
    GET  /info     model metadata (name, params, metrics)
    POST /predict  one or more patients -> prediction + confidence
    GET  /metrics  Prometheus metrics
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))


# ---------- structured JSON logging ----------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": self.formatTime(record), "level": record.levelname,
                   "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("heart_api")
logger.handlers = [handler]
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.propagate = False


# ---------- Prometheus metrics ----------
REQUESTS = Counter("api_requests_total", "HTTP requests",
                   ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency",
                    ["path"], buckets=(.005, .01, .025, .05, .1, .25, .5, 1))
PREDICTIONS = Counter("model_predictions_total", "Predictions by class",
                      ["label"])
CONFIDENCE = Histogram("model_prediction_confidence", "Confidence of predictions",
                       buckets=(.5, .6, .7, .8, .9, .95, 1.0))


# ---------- request / response schemas ----------
class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[63])
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="chest pain type")
    trestbps: float = Field(..., gt=0, le=300, description="resting BP (mm Hg)")
    chol: float = Field(..., gt=0, le=700, description="serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2]
    thalach: float = Field(..., gt=0, le=250, description="max heart rate")
    exang: Literal[0, 1] = Field(..., description="exercise-induced angina")
    oldpeak: float = Field(..., ge=-5, le=10, description="ST depression")
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3] | None = Field(None, description="major vessels (0-3)")
    thal: Literal[3, 6, 7] | None = Field(None, description="3 normal, 6 fixed, 7 reversible")

    model_config = {"json_schema_extra": {"examples": [{
        "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
        "ca": 0, "thal": 6}]}}


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float


class PredictResponse(BaseModel):
    model: str
    results: list[Prediction]


# ---------- app ----------
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    meta_path = MODEL_DIR / "model_meta.json"
    state["meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    logger.info("model loaded", extra={"extra_fields": {
        "model": state["meta"].get("model_name"), "dir": str(MODEL_DIR)}})
    yield
    state.clear()


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        if path != "/metrics":  # don't flood logs with scrapes
            logger.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method,
                "path": path, "status": status,
                "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/info")
def info():
    return state.get("meta", {})


@app.post("/predict", response_model=PredictResponse)
def predict(patients: Patient | list[Patient]):
    batch = patients if isinstance(patients, list) else [patients]
    if not batch:
        raise HTTPException(status_code=422, detail="empty input")

    features = state["meta"].get("features")
    X = pd.DataFrame([p.model_dump() for p in batch])
    if features:
        X = X[features]
    X = X.astype(float)  # None -> NaN, handled by the pipeline's imputers

    proba = state["model"].predict_proba(X)[:, 1]
    results = []
    for p in proba:
        pred = int(p >= 0.5)
        conf = float(p if pred else 1 - p)
        label = "disease" if pred else "no_disease"
        PREDICTIONS.labels(label).inc()
        CONFIDENCE.observe(conf)
        results.append(Prediction(prediction=pred, label=label,
                                  confidence=round(conf, 4),
                                  probability_disease=round(float(p), 4)))
    logger.info("prediction", extra={"extra_fields": {
        "n": len(results), "labels": [r.label for r in results]}})
    return PredictResponse(model=state["meta"].get("model_name", "unknown"),
                           results=results)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
