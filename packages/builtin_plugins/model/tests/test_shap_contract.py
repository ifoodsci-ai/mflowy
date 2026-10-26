"""#85 shap 副产物契约：写→盘→读 round-trip 真跑 + 注记缺失 fail-loud + 前缀字面量单源扫描。

契约单源住 ``shap/contract``（前缀/记账列/注记三元组/分块迭代/宽表组装）——
写方（log_shap_explanation 中间件）、读方（shap__load）、生产方
（shap__explanation / shap__load）三方共享，本文件钉住三方同形。
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shap
from mflowy.model.middlewares.log_shap_explanation import log_shap_explanation
from mflowy.model.plot.model_interpretability.shap.contract import (
    CAT_PREFIX,
    SHAP_PREFIX,
    VALUE_PREFIX,
    annotation_triple,
    attach_annotations,
    explanation_to_frame,
    iter_output_blocks,
)
from mflowy.model.plot.model_interpretability.shap.load import load as shap_load


def _make_explanation(ndim: int = 2) -> shap.Explanation:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(6, 3)) if ndim == 2 else rng.normal(size=(6, 3, 2))
    exp = shap.Explanation(values=values, data=rng.normal(size=(6, 3)), feature_names=["f0", "f1", "f2"])
    return attach_annotations(
        exp, y_name="y", categorical_features=["f1"], classes=["cls0", "cls1"] if ndim == 3 else ["y"]
    )


@pytest.fixture(autouse=True)
def isolate_tracking(tmp_path, monkeypatch):
    import mlflow

    uri = f"sqlite:///{tmp_path}/tracking.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)


def _roundtrip(exp: shap.Explanation) -> None:
    import mlflow
    from mflowy.utils.mlflow import download_run_artifact, run_artifacts

    with mlflow.start_run() as run:
        out = log_shap_explanation(None, lambda _ctx: exp)
    assert out is exp  # 中间件透传产物
    assert "shap_values.parquet" in run_artifacts(run.info.run_id)
    df = pd.read_parquet(download_run_artifact(run.info.run_id, "shap_values.parquet"))
    rebuilt = shap_load(load=[df])
    assert np.allclose(rebuilt.values, exp.values) and np.allclose(np.asarray(rebuilt.data), np.asarray(exp.data))
    assert list(rebuilt.feature_names) == list(exp.feature_names)
    assert annotation_triple(rebuilt) == annotation_triple(exp)  # 注记三元组同形


def test_roundtrip_write_disk_read_2d():
    """写→盘→读真跑（2D）：middleware 落 parquet 工件 → shap__load 重建与原件同形。"""
    _roundtrip(_make_explanation(2))


def test_roundtrip_write_disk_read_3d():
    """3D 多分类：按类分块落盘 → 重建 stack 回 3D，classes 序保真。"""
    _roundtrip(_make_explanation(3))


def test_missing_annotations_fail_loud():
    """裸 Explanation 缺注记三元组 → fail-loud 带产出面指引（不再 '-' 占位静默回退）。"""
    bare = shap.Explanation(values=np.zeros((3, 2)), data=np.zeros((3, 2)), feature_names=["a", "b"])
    with pytest.raises(ValueError, match="契约属性注记"):
        explanation_to_frame(bare)
    with pytest.raises(ValueError, match="契约属性注记"):
        list(iter_output_blocks(bare))


def test_annotated_blocks_iterate_in_class_order():
    """完整注记正常出块：3D 块序 = classes 序（写方分块与重建 stack 同一序）。"""
    exp = _make_explanation(3)
    assert [label for label, _ in iter_output_blocks(exp)] == ["cls0", "cls1"]


def test_prefix_literals_single_source():
    """#85 单源红线：三前缀字面量全仓包源码唯一（contract.py 常量定义处）。"""
    import mflowy.model.plot.model_interpretability.shap.contract as contract_mod

    root = Path(contract_mod.__file__).parents[8]
    sources = [p for p in sorted(root.joinpath("packages").rglob("*.py")) if "tests" not in p.parts]
    for prefix in (SHAP_PREFIX, VALUE_PREFIX, CAT_PREFIX):
        occurrences = [str(p.relative_to(root)) for p in sources if f'"{prefix}"' in p.read_text(encoding="utf-8")]
        assert occurrences == [
            "packages/builtin_plugins/model/mflowy/model/plot/model_interpretability/shap/contract.py"
        ], (prefix, occurrences)
