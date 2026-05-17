from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from predict import predict_biogas

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html", encoding="utf-8").read()


@app.post("/predict")
def predict(data: DigestorInput):
    try:
        return predict_biogas(data.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))