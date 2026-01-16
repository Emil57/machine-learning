import os
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from f1.feature_engineering import train_te, train, valid_te, valid
from f1 import MODELS_DIR

try:
    raw_model = SGDClassifier( 
        loss="log_loss", # logistic regression 
        max_iter=1000, # number of iterations 
        class_weight={0: 1.0, 1: 30}, # handle imbalance 
        random_state=42,
        )
    raw_model.fit(train_te, train["won"])

    # 11) Probability calibration (Platt scaling on validation) 
    model_calibrated = CalibratedClassifierCV(estimator=raw_model, method="sigmoid", cv="prefit")
    model_calibrated.fit(valid_te, valid["won"])

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model_calibrated, f'{MODELS_DIR}/model.pkl')
    print(f"Saved model to {MODELS_DIR}")
except Exception as e:
    print(f'Error in feature engineering: {e}')
    raise