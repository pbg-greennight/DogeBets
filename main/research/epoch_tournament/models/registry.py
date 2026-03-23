from __future__ import annotations

from .model_logreg import LogRegEpochModel
from .model_rf import RFEpochModel
from .model_xgb_fallback import XGBFallbackEpochModel
from .model_rules import StructureRuleModel


MODEL_REGISTRY = {
    "logreg": LogRegEpochModel,
    "rf": RFEpochModel,
    "xgb_fallback": XGBFallbackEpochModel,
    "rules": StructureRuleModel,
}


def build_model(model_cfg: dict):
    clf_name = model_cfg["classifier"]
    if clf_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown classifier: {clf_name}")
    return MODEL_REGISTRY[clf_name](model_cfg)