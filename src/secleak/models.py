"""Fixed model zoo. Preprocessing is fit on TRAIN only (inside the sklearn Pipeline)."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Small, identical search space per model; same budget under every protocol.
GRIDS = {
    "logreg": [{"clf__C": c} for c in (0.1, 1.0, 10.0)],
    "rf": [{"clf__max_depth": d} for d in (8, 16, None)],
    "gbdt": [{"clf__learning_rate": lr} for lr in (0.05, 0.1, 0.2)],
    "mlp": [{"clf__alpha": a} for a in (1e-4, 1e-2)],
}


def _prep(scale: bool):
    num = make_pipeline(SimpleImputer(strategy="median"),
                        StandardScaler() if scale else "passthrough")
    cat = make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                        OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    return ColumnTransformer([
        ("num", num, make_column_selector(dtype_include=np.number)),
        ("cat", cat, make_column_selector(dtype_exclude=np.number)),
    ])


def make_model(name: str, seed: int = 0, n_jobs: int = 1) -> Pipeline:
    if name == "logreg":
        clf, scale = LogisticRegression(max_iter=500, random_state=seed), True
    elif name == "rf":
        clf, scale = RandomForestClassifier(n_estimators=150, n_jobs=n_jobs, random_state=seed), False
    elif name == "gbdt":
        clf, scale = HistGradientBoostingClassifier(random_state=seed), False
    elif name == "mlp":
        clf, scale = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=60, random_state=seed), True
    elif name == "xgb":
        from xgboost import XGBClassifier  # optional dependency
        clf, scale = XGBClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed,
                                   eval_metric="logloss"), False
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("prep", _prep(scale)), ("clf", clf)])


MODEL_NAMES = ["logreg", "rf", "gbdt", "mlp"]
