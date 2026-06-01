"""Central configuration: one source of truth for paths, the random seed,
hyperparameters and run settings.

Precedence (low → high):  dataclass defaults  <  conf/config.yaml  <  env vars.
That's the 12-factor pattern: sane defaults in code, project values in a file,
and deployment-specific overrides (e.g. the prod MLflow URL) from the environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Compute the repo root from THIS file, not the current directory, so paths
# resolve correctly no matter where you run the code from.
REPO_ROOT = Path(__file__).resolve().parents[2]   # src/quant_risk/config.py -> repo root
DEFAULT_CONF = REPO_ROOT / "conf" / "config.yaml"


@dataclass
class Paths:
    root: Path = REPO_ROOT
    data: Path = REPO_ROOT / "data"
    bronze: Path = REPO_ROOT / "data" / "bronze"
    silver: Path = REPO_ROOT / "data" / "silver"
    gold: Path = REPO_ROOT / "data" / "gold"

    def make(self) -> None:
        for p in (self.data, self.bronze, self.silver, self.gold):
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelParams:
    seed: int = 42
    test_size: float = 0.2
    max_iter: int = 1000      # logistic-regression solver iterations
    auc_gate: float = 0.60    # minimum AUC required to register/promote a model


@dataclass
class Config:
    seed: int = 42
    n_synthetic: int = 50_000              # rows to generate when there's no real file
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "credit-pd"
    model_name: str = "credit_pd"          # registry name — for the task, not the algorithm
    paths: Paths = field(default_factory=Paths)
    model: ModelParams = field(default_factory=ModelParams)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    conf_path = Path(path or os.getenv("QR_CONFIG", DEFAULT_CONF))
    if conf_path.exists():
        raw = yaml.safe_load(conf_path.read_text()) or {}
        cfg.seed = raw.get("seed", cfg.seed)
        cfg.n_synthetic = raw.get("n_synthetic", cfg.n_synthetic)
        cfg.mlflow_tracking_uri = raw.get("mlflow_tracking_uri", cfg.mlflow_tracking_uri)
        cfg.mlflow_experiment = raw.get("mlflow_experiment", cfg.mlflow_experiment)
        cfg.model_name = raw.get("model_name", cfg.model_name)
        for k, v in (raw.get("model") or {}).items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)
    # env overrides — deployment-specific (e.g. point at a real MLflow server)
    cfg.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg.mlflow_tracking_uri)
    cfg.model.seed = cfg.seed
    return cfg