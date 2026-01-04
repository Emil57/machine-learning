import os
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from f1.feature_engineering import train_te, train, valid_te, valid

OUTPUT_DIR = os.path.join(
        os.path.abspath("../../"), 
        "artifacts", 
        "f1", 
        "models",
    )

try:
    clf = SGDClassifier( 
        loss="log_loss", # logistic regression 
        max_iter=1000, # number of iterations 
        class_weight={0: 1.0, 1: 20.0}, # handle imbalance 
        random_state=42,
        )
    clf.fit(train_te, train["won"])

    # 11) Probability calibration (Platt scaling on validation) 
    calibrated = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
    calibrated.fit(valid_te, valid["won"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(calibrated, f'{OUTPUT_DIR}/model.pkl')
    print(f"Saved model to {OUTPUT_DIR}")
except Exception as e:
    print(f'Error in feature engineering: {e}')
    raise