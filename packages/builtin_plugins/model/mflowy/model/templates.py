"""建模域模板常量 + 步骤片段门 + model 步计数（tools 与 job_provider 共用）。"""

from __future__ import annotations

from pathlib import Path

import yaml

TEMPLATES_DIR = Path(__file__).parent / "templates"
MODELING_TEMPLATE = TEMPLATES_DIR / "modeling.yaml.j2"
HYPERPARAMETERS_SEARCH_TEMPLATE = TEMPLATES_DIR / "hyperparameters_search.yaml.j2"
EXPLANATION_TEMPLATE = TEMPLATES_DIR / "explanation.yaml.j2"
PREDICT_TEMPLATE = TEMPLATES_DIR / "predict.yaml.j2"
INVERSE_OPTIMIZATION_TEMPLATE = TEMPLATES_DIR / "inverse_optimization.yaml.j2"


def parse_steps_fragment(steps_text: str) -> list:
    """步骤片段门（单一谓词，#83）：yaml 明文 → 顶层步骤列表。

    形状不符（空明文 / 顶层 mapping 完整文档 / 非列表）raise ValueError 带
    修复指引（name/description/flow 外壳由建模模板自动包装）。MCP 预检面
    （validate_modeling_steps）与提交侧（LocalJobProvider._require_steps）共享
    本谓词——包装差异（``valid=False`` 结构化返回 vs raise fail-loud）留在各自面。
    """
    if not steps_text or not steps_text.strip():
        raise ValueError("modeling_steps 明文为空——须为顶层步骤列表片段（- name: 开头）")
    parsed = yaml.safe_load(steps_text)
    if isinstance(parsed, dict):
        raise ValueError(
            "modeling_steps 须为顶层步骤列表片段（- name: 开头的 YAML list）——"
            "name/description/flow 外壳由建模模板自动包装，请剔除顶层外壳键"
        )
    if not isinstance(parsed, list):
        raise ValueError("modeling_steps 须为顶层步骤列表片段（- name: 开头的 YAML list）")
    return parsed


def count_model_steps(steps_yaml: str) -> int:
    """统计 modeling_steps 中 namespace=model 的步骤数（平铺列表，#59）。"""
    parsed = yaml.safe_load(steps_yaml)
    if not isinstance(parsed, list):
        return 0
    return sum(1 for s in parsed if isinstance(s, dict) and s.get("namespace") == "model")
