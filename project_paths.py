"""
Project paths for training.ipynb and prediction.ipynb (local conda env).

Set IPL_PROJECT_ROOT if Jupyter's working directory is not the project folder.
Example (PowerShell): $env:IPL_PROJECT_ROOT = 'D:/Projects/IPL 2026'
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("IPL_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent


ROOT = project_root()
DATASET_DIR = ROOT / "Datasets"
MODEL_DIR = ROOT / "Models"
