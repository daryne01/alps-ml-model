import numpy as np
import joblib

RF_WEIGHT  = 0.60
XGB_WEIGHT = 0.40

rf_model  = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
le        = joblib.load('models/label_encoder.pkl')


def predict_biogas(input_data: dict) -> dict:
    feedstock_encoded = le.transform([input_data['feedstock']])[0]

    X = np.array([[
        feedstock_encoded,
        input_data.get('daily_input', np.nan),
        input_data.get('ts', np.nan),
        input_data.get('vs_ts', np.nan),
        input_data.get('olr', np.nan),
        input_data.get('temperature', np.nan),
        input_data.get('hrt', np.nan),
        input_data.get('volume', np.nan),
        input_data.get('ph', np.nan),
        input_data.get('ammonia', np.nan),
        input_data.get('vfa_alk', np.nan),
    ]])

    rf_pred  = rf_model.predict(X)[0]
    xgb_pred = xgb_model.predict(X)[0]
    ensemble = (RF_WEIGHT * rf_pred) + (XGB_WEIGHT * xgb_pred)

    tree_preds = np.array([tree.predict(X)[0] for tree in rf_model.estimators_])
    lower = np.percentile(tree_preds, 10)
    upper = np.percentile(tree_preds, 90)

    return {
        'predicted_yield_m3_per_tonne': round(float(ensemble), 1),
        'confidence_range': {
            'lower': round(float(lower), 1),
            'upper': round(float(upper), 1),
        },
        'rf_prediction':  round(float(rf_pred), 1),
        'xgb_prediction': round(float(xgb_pred), 1),
    }


if __name__ == '__main__':
    result = predict_biogas({
        'feedstock':   'Food waste',
        'daily_input': 60,
        'ts':          22,
        'vs_ts':       85,
        'olr':         4.8,
        'temperature': 38,
        'hrt':         18,
        'volume':      2000,
        'ph':          7.6,
        'ammonia':     3400,
        'vfa_alk':     0.48,
    })
    print(result)
