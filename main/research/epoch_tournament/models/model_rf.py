from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseEpochModel


class RFEpochModel(BaseEpochModel):
    def __init__(self, model_cfg: dict):
        super().__init__(model_cfg)
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    def fit(self, x, y):
        self.model.fit(x, y)
        self._is_fit = True

    def predict_proba(self, x):
        return self.model.predict_proba(x)