# ALPS Ecoscience — Biogas Yield Prediction Model

A machine learning model that predicts biogas yield from anaerobic digestion plants using feedstock and operating data.

---

## What it does

Takes in plant operating data (feedstock type, temperature, pH, ammonia levels etc.) and predicts how much biogas the plant should be producing. If the actual yield is lower than predicted, something is wrong.

---

## Why Random Forest + XGBoost

Two models are combined because neither is perfect on its own.

**Random Forest** is reliable and handles missing data well. It also tells you which inputs matter most — useful for explaining to a client why their yield is low.

**XGBoost** is better at picking up on combined problems — for example when high ammonia and short retention time are both happening at the same time. One model on its own would miss that.

Combined they are more accurate than either alone. The final prediction is 60% Random Forest and 40% XGBoost.

---

## What data it needs

One row per plant observation. The more rows the better — 50 minimum, 200+ ideal.

| Column | Example |
|---|---|
| feedstock | Food waste |
| daily_input | 60 |
| ts | 22 |
| vs_ts | 85 |
| olr | 4.8 |
| temperature | 38 |
| hrt | 18 |
| volume | 2000 |
| ph | 7.6 |
| ammonia | 3400 |
| vfa_alk | 0.48 |
| biogas_yield | 72 |
| methane_pct | 52 |

Blank cells are fine — the model fills gaps automatically.

---

## How to run it

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
python train_model.py
```

Start the API:
```bash
uvicorn api:app --reload --port 8000
```

---

## Files

```
alps-ml-model/
├── data/plant_observations.csv   ← training data
├── models/                       ← saved models go here after training
├── train_model.py                ← builds the model
├── predict.py                    ← runs predictions
├── api.py                        ← API endpoint
└── requirements.txt
```

---

## Next steps

- Add more real plant data from ALPS clients
- Deploy API online so the frontend can call it live
- Upgrade to LSTM once we have 12+ months of daily data per plant

---

*ALPS Ecoscience*
