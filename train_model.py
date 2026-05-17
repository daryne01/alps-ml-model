import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

FEATURES = [
    'feedstock_encoded', 'daily_input', 'ts', 'vs_ts', 'olr',
    'temperature', 'hrt', 'volume', 'ph', 'ammonia', 'vfa_alk'
]
RF_WEIGHT  = 0.60
XGB_WEIGHT = 0.40


def train(target: str = 'biogas_yield', data_path: str = 'data/plant_observations.csv'):
    df = pd.read_csv(data_path)

    le = LabelEncoder()
    df['feedstock_encoded'] = le.fit_transform(df['feedstock'])

    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("Random Forest  R²:", round(r2_score(y_test, rf_pred), 4))
    print("Random Forest MAE:", round(mean_absolute_error(y_test, rf_pred), 2))

    xgb_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    print("XGBoost  R²:", round(r2_score(y_test, xgb_pred), 4))
    print("XGBoost MAE:", round(mean_absolute_error(y_test, xgb_pred), 2))

    ensemble_pred = (RF_WEIGHT * rf_pred) + (XGB_WEIGHT * xgb_pred)
    print("\nEnsemble  R²:", round(r2_score(y_test, ensemble_pred), 4))
    print("Ensemble MAE:", round(mean_absolute_error(y_test, ensemble_pred), 2))

    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
    print(f"\nCross-validation R² (5-fold): {round(cv_scores.mean(), 4)} ± {round(cv_scores.std(), 4)}")

    importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature Importances (Random Forest):")
    print(importances.round(4))

    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    joblib.dump(xgb_model, 'models/xgb_model.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    print("\nModels saved to models/")

    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=FEATURES, show=False)
        plt.tight_layout()
        plt.savefig('models/shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("SHAP summary saved to models/shap_summary.png")
    except Exception as e:
        print(f"SHAP plot skipped: {e}")


if __name__ == '__main__':
    train()
