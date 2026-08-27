"""模块可用性标注：模块路径 → 所属 extra（stats / modeling）。

目录即边界（与 hatch_metadata._STEP_OF_DIR 同风格）；plots 族按二级目录细分
（model_interpretability 依赖 shap 属 [modeling]）。纯 dict 无第三方依赖：
构建期由 hatch_metadata.py 加载校验「每个插件都有标注」（漏配即构建失败），
运行期由 mcp 层零成本标注目录与 get_module_info——base 环境装不上数据栈，
但能看清装哪个 extra 可用。
"""

_EXTRA_OF_DIR = {
    "loaders": "stats",
    "cleaners": "stats",
    "x_y": "stats",
    "statistic": "stats",
    "plots/data_analysis": "stats",
    "plots/model_evaluation": "stats",
    "x_transformer": "modeling",
    "cross_validation": "modeling",
    "model": "modeling",
    "plots/model_interpretability": "modeling",
}


def extra_of(module_path: str) -> str | None:
    """entry point value 的模块路径（如 ``mflowy.builtin_plugins.plots.model_interpretability.shap.summary:shap_summary``）→ 所属 extra；最长目录前缀匹配，非本包路径返 None。"""
    parts = module_path.split(":")[0].split(".")
    try:
        rest = parts[parts.index("builtin_plugins") + 1 :]
    except ValueError:
        return None
    for n in range(len(rest), 0, -1):
        if hit := _EXTRA_OF_DIR.get("/".join(rest[:n])):
            return hit
    return None
