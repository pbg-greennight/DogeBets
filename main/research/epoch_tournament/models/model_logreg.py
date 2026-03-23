from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base_model import BaseEpochModel


class LogRegEpochModel(BaseEpochModel):
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        )

    def fit(self, x, y):
        self.model.fit(x, y)
        self._is_fit = True

    def predict_proba(self, x):
        return self.model.predict_proba(x)