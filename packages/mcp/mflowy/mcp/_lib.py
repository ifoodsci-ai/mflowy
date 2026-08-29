"""mcp 层内部共享件：模板路径常量 + model 步计数 + ``py:target`` 引用解析。

tools.py 与 job_provider/local.py 同用（分析工具本地执行不经 JobProvider，
但编排同形）；下划线开头标明非 MCP 工具，runner 不经 getattr 暴露。
"""

from __future__ import annotations

from pathlib import Path

import yaml
from mflowy.driver.config import iter_step_dicts
from mflowy.utils.path import set_task_dir, split_path_to_py_with_target

TEMPLATES_DIR = Path(__file__).parent / "templates"
DATA_PROFILE_TEMPLATE = TEMPLATES_DIR / "data_profile.yaml.j2"
EDA_TEMPLATE = TEMPLATES_DIR / "eda.yaml.j2"
INFER_TASK_TYPE_TEMPLATE = TEMPLATES_DIR / "infer_task_type.yaml.j2"
MODELING_TEMPLATE = TEMPLATES_DIR / "modeling.yaml.j2"
EXPLANATION_TEMPLATE = TEMPLATES_DIR / "explanation.yaml.j2"
PREDICT_TEMPLATE = TEMPLATES_DIR / "predict.yaml.j2"
INVERSE_OPTIMIZATION_TEMPLATE = TEMPLATES_DIR / "inverse_optimization.yaml.j2"


def count_model_steps(steps_yaml: str) -> int:
    """统计 modeling_steps 中 type=model 的步骤数（递归 branches/steps）。"""
    parsed = yaml.safe_load(steps_yaml)
    if not isinstance(parsed, list):
        return 0
    return sum(1 for s in iter_step_dicts(parsed) if s.get("type") == "model")


def resolve_data_ref(data: str) -> tuple[Path, str]:
    """``path:func`` 引用 → (路径, 绝对引用串)；路径经 set_task_dir 绑定任务目录。"""
    path, func = split_path_to_py_with_target(data)
    set_task_dir(path)
    ref = path.absolute().as_posix()
    if func:
        ref = f"{ref}:{func}"
    return path, ref
