"""建模域模板常量 + model 步计数（tools 与 job_provider 共用）。"""

from __future__ import annotations

from pathlib import Path

import yaml

TEMPLATES_DIR = Path(__file__).parent / "templates"
MODELING_TEMPLATE = TEMPLATES_DIR / "modeling.yaml.j2"
HYPERPARAMETERS_SEARCH_TEMPLATE = TEMPLATES_DIR / "hyperparameters_search.yaml.j2"
EXPLANATION_TEMPLATE = TEMPLATES_DIR / "explanation.yaml.j2"
PREDICT_TEMPLATE = TEMPLATES_DIR / "predict.yaml.j2"
INVERSE_OPTIMIZATION_TEMPLATE = TEMPLATES_DIR / "inverse_optimization.yaml.j2"


def count_model_steps(steps_yaml: str) -> int:
    """统计 modeling_steps 中 namespace=model 的步骤数（平铺列表，#59）。"""
    parsed = yaml.safe_load(steps_yaml)
    if not isinstance(parsed, list):
        return 0
    return sum(1 for s in parsed if isinstance(s, dict) and s.get("namespace") == "model")
