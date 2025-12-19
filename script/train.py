from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# script/train.py -> parent=script -> parent.parent=ROOT
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
ART_DIR = ROOT / "artifacts"
TARGET = "Revenue"
SEED = 42

# 컬럼 정의(원하면 core/schema로 이동)
NUMERIC = [
    "Administrative","Administrative_Duration","Informational","Informational_Duration",
    "ProductRelated","ProductRelated_Duration","BounceRates","ExitRates","PageValues","SpecialDay",
]
CATEGORICAL = ["Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend"]

def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )

def build_model(use_smote: bool):
    """
    불균형 보정:
      - 기본: class_weight="balanced"
      - 선택: SMOTE (imbalanced-learn 설치 시)
    """
    pre = build_preprocessor()

    base_lr = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",   # 옵션 A: 기본 보정(강추)
        random_state=SEED,
    )

    # 확률(%)이 중요하니 calibration을 기본으로
    calibrated = CalibratedClassifierCV(
        estimator=base_lr,
        method="sigmoid",
        cv=5
    )

    if not use_smote:
        return Pipeline([("preprocess", pre), ("model", calibrated)])

    # 옵션 B: SMOTE는 train에만 적용되어야 하므로 imblearn 파이프라인 사용
    try:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=SEED, k_neighbors=5)
        return ImbPipeline([("preprocess", pre), ("smote", smote), ("model", calibrated)])
    except Exception as e:
        print("[WARN] SMOTE를 사용하려면 `pip install imbalanced-learn`가 필요합니다.")
        print("[WARN] 현재는 class_weight만 적용해서 학습합니다. (원인:", repr(e), ")")
        return Pipeline([("preprocess", pre), ("model", calibrated)])

def load_split():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    y_train = train_df[TARGET].astype(int)
    X_train = train_df.drop(columns=[TARGET])

    y_test = test_df[TARGET].astype(int)
    X_test = test_df.drop(columns=[TARGET])

    return X_train, y_train, X_test, y_test

def main():
    ART_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_split()

    # ===== 여기서 옵션 선택 =====
    USE_SMOTE = True   # 설치돼 있으면 SMOTE까지, 아니면 자동으로 class_weight만 사용
    pipe = build_model(use_smote=USE_SMOTE)

    pipe.fit(X_train, y_train)

    # 간단 평가(훈련 중 빠른 sanity check)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "use_smote_requested": bool(USE_SMOTE),
        "test_positive_rate": float(y_test.mean()),
    }

    joblib.dump(pipe, ART_DIR / "purchase_model.joblib")
    (ART_DIR / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("Saved:", ART_DIR / "purchase_model.joblib")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
