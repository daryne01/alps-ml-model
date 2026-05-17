from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from predict import predict_biogas

app = FastAPI(title="ALPS Biogas Predictor API")


class DigestorInput(BaseModel):
    feedstock:   str
    daily_input: Optional[float] = None
    ts:          Optional[float] = None
    vs_ts:       Optional[float] = None
    olr:         Optional[float] = None
    temperature: Optional[float] = None
    hrt:         Optional[float] = None
    volume:      Optional[float] = None
    ph:          Optional[float] = None
    ammonia:     Optional[float] = None
    vfa_alk:     Optional[float] = None


@app.post("/predict")
def predict(data: DigestorInput):
    try:
        return predict_biogas(data.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
