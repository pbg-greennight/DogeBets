from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def save_model_artifact(model: Any, feature_cols: list[str], model_cfg: dict, config: dict, artifact_dir: Path, artifact_name: str) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / f"{artifact_name}.pkl"
    config_path = artifact_dir / f"{artifact_name}_config.json"
    meta_path = artifact_dir / f"{artifact_name}_metadata.json"

    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "model_cfg": model_cfg,
        "config": config,
    }

    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metadata = {
        "artifact_name": artifact_name,
        "model_path": str(model_path),
        "config_path": str(config_path),
        "feature_count": len(feature_cols),
        "model_id": model_cfg.get("model_id", "unknown"),
        "family": model_cfg.get("family", "unknown"),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_model_artifact(model_path: str | Path) -> dict:
    model_path = Path(model_path)
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    return payload