"""Model serialization and registry for versioned model tracking."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


class ModelSerializer:
    """Serialize trained models to disk with metadata."""

    def __init__(self, artifact_dir: str | Path = "artifacts/models") -> None:
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        name: str,
        model_type: str,
        metrics: dict[str, Any] | None = None,
        hyperparams: dict[str, Any] | None = None,
        dataset_hash: str | None = None,
    ) -> tuple[str, Path]:
        """
        Save model with metadata.

        Args:
            model: trained sklearn/xgb/prophet object.
            name: model identifier (e.g., 'churn_xgboost').
            model_type: 'logistic_regression', 'xgboost', 'prophet', etc.
            metrics: dict of evaluation metrics.
            hyperparams: dict of model hyperparameters.
            dataset_hash: MD5 of training data for reproducibility.

        Returns:
            (version_id, filepath) tuple.
        """
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.artifact_dir / f"{name}_{version_id}.pkl"

        # Serialize model
        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        # Save metadata alongside
        metadata = {
            "name": name,
            "type": model_type,
            "version": version_id,
            "timestamp": datetime.now().isoformat(),
            "filepath": str(filepath),
            "metrics": metrics or {},
            "hyperparams": hyperparams or {},
            "dataset_hash": dataset_hash,
            "file_size_mb": round(filepath.stat().st_size / (1024**2), 4),
        }
        meta_path = filepath.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return version_id, filepath


class ModelRegistry:
    """Central manifest for all versioned models."""

    def __init__(self, manifest_file: str | Path = "artifacts/model_manifest.json") -> None:
        self.manifest_file = Path(manifest_file)
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, Any] = self._load_manifest()

    def register(
        self,
        name: str,
        filepath: Path | str,
        version: str,
        metrics: dict[str, Any],
        hyperparams: dict[str, Any],
        model_type: str,
        dataset_hash: str | None = None,
    ) -> None:
        """Register a model version in the manifest."""
        if name not in self.models:
            self.models[name] = {"versions": {}}

        self.models[name]["versions"][version] = {
            "filepath": str(filepath),
            "metrics": metrics,
            "hyperparams": hyperparams,
            "model_type": model_type,
            "dataset_hash": dataset_hash,
            "timestamp": datetime.now().isoformat(),
        }
        self.models[name]["latest"] = version
        self._save_manifest()

    def load(self, name: str, version: str = "latest") -> Any:
        """Load a model by name and version."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")

        versions = self.models[name]["versions"]
        if version == "latest":
            version = max(versions.keys())

        if version not in versions:
            raise ValueError(f"Version '{version}' not found for model '{name}'")

        filepath = versions[version]["filepath"]
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def get_metrics(self, name: str, version: str = "latest") -> dict[str, Any]:
        """Retrieve model metrics for a specific version."""
        if version == "latest":
            version = max(self.models[name]["versions"].keys())
        return self.models[name]["versions"][version]["metrics"]

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self.models.keys())

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model."""
        if name not in self.models:
            return []
        return sorted(self.models[name]["versions"].keys())

    def _load_manifest(self) -> dict[str, Any]:
        if self.manifest_file.exists():
            with open(self.manifest_file, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        with open(self.manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.models, f, indent=2, default=str)


def compute_dataset_hash(data: Any) -> str:
    """Compute an MD5 hash of a DataFrame for reproducibility tracking."""
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        content = data.to_json().encode()
    else:
        content = str(data).encode()
    return hashlib.md5(content).hexdigest()
