"""#66 终局观察面：OOF 图集 + draw_taylor_diagram + report_skeleton + finish(pdf) 单动作。

- OOF 图集：run_hpo 的观察面指向 OOF 源（cross_val_predict × train 侧 y_true），
  产物挂 run 下（run_id 寻址）可列出
- draw_taylor_diagram（改名自 taylor_diagram_report，BREAKING）：多终训 run 副产物
  fan-in → (corr, σ, rmse) + 图落盘；y_test 逐位同一断言（不同 holdout 进同一张
  对比图 = 撒谎的图——拒绝）；只绘图不做报告
- modeling_report_skeleton：只读零副作用——表格数字从账本渲染（逐位相等）、
  图片清单（OOF 图集/副产物指针/泰勒图路径）、叙事槽样板
- finish(pdf) 单动作（BREAKING）：pdf 必填、收编进 deliverables（tmp/rename 原子
  落盘）+ verdict；重入幂等（交付报告不可变更）；preview 分支退役
- #92 重栈纪律：终态会话 fixture 模块级共享（只读族一次构建）；终训副产物按
  run_id 跨用例读回——tracking 库随之模块级共享
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mflowy.driver.mdp_session import verify_verdict
from mflowy.model import session as ms
from mflowy.model.tools import draw_taylor_diagram, modeling_finish, modeling_report_skeleton

_K = "RF|passthrough|passthrough"
_K2 = "XGB|passthrough|passthrough"


@pytest.fixture(scope="module")
def sessions_root(tmp_path_factory):
    """模块级会话根（#92）：重栈会话只建一遍——function 作用域曾致 9 用例 × 4 管线
    线性重建、单文件峰值 10GB（全量 ~20GB）。"""
    root = tmp_path_factory.mktemp("modeling_sessions")
    mp = pytest.MonkeyPatch()
    mp.setenv("MFLOWY_MODELING_SESSIONS", str(root))
    yield root
    mp.undo()


@pytest.fixture(scope="module")
def reg_csv(tmp_path_factory) -> str:
    n = 120
    rng = np.random.default_rng(4)
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "target": 2.0 * x + 0.1 * rng.normal(size=n)})
    p = tmp_path_factory.mktemp("reg") / "reg.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture(scope="module")
def shared_tracking(tmp_path_factory):
    """模块级共享 tracking 库（#92）：终训副产物按 run_id 跨用例读回
    （taylor_for_runs → MlflowClient），寻址须同库——per-test 换库切断 fan-in。

    fluent 全局随建随还：``set_tracking_uri`` 遮蔽 env（进程级残留会泄进后续
    文件的 env-only 隔离）；store 须在 cwd=tmp 时急切建——sqlite 库的工件根
    ``./mlruns`` 按 store 初始化时刻的 cwd 绝对化（仓库根零运行产物）。"""
    import mlflow

    d = tmp_path_factory.mktemp("tracking")
    uri = f"sqlite:///{d}/tracking.db"
    prev_cwd = os.getcwd()
    os.chdir(d)
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.MlflowClient().search_experiments()  # 急切建库 + Default 实验（工件根锚 tmp）
        yield uri
    finally:
        os.chdir(prev_cwd)
        mlflow.set_tracking_uri(None)  # 复位 env 驱动解析（受支持的复位面）


@pytest.fixture(autouse=True)
def _pin_shared_tracking(monkeypatch, shared_tracking):
    """共享库复指（晚于根 conftest 的 per-test 隔离生效）：读回缝
    ``run_artifacts → setup()`` 从 env 解析 tracking 再重设 fluent——env 不钉
    共享库，终训副产物的跨用例 fan-in 寻址即断。"""
    import mlflow

    monkeypatch.setenv("MLFLOW_TRACKING_URI", shared_tracking)
    mlflow.set_tracking_uri(shared_tracking)


CARD = {"domain": "测试", "target": "target"}


CV_CARD = "cv: 3"


def _start(root, csv, fams=("RF",)):
    return ms.start_session(
        csv,
        CV_CARD,
        card=CARD,
        task="regression",
        metric="mae",
        max_steps=8,
        candidates=[{"family": f, "baseline": i == 0} for i, f in enumerate(fams)],
    )


def _manifest(root, sid):
    return json.loads((root / sid / "manifest.json").read_text(encoding="utf-8"))


def _pdf(tmp_path, name="final_report.pdf") -> Path:
    p = tmp_path / name
    p.write_bytes(b"%PDF-1.4 pi-authored")
    return p


def _build_delivered(root, csv):
    """双模型终态会话：RF + XGB 各搜一次并 final_train_test（4 条管线）。"""
    s = _start(root, csv, fams=("RF", "XGB"))
    sid = s["session_id"]
    for fam, params in (("RF", {"n_estimators": 15}), ("XGB", {"n_estimators": 15})):
        ms.apply_action(sid, "select_model", {"family": fam})
        ms.apply_action(sid, "set_hp", params)  # 收敛 trial 成本
        ms.apply_action(sid, "run_hpo", {"n_trials": 2})
        ms.apply_action(sid, "final_train_test")
    return s


@pytest.fixture(scope="module")
def delivered(sessions_root, reg_csv, shared_tracking):
    """模块级共享双模型终态会话（#92）——共享前提 = 消费用例只读：不改光标、
    不 apply、不 finish（finish 终态化 manifest）。写面用例走 delivered_fresh /
    delivered_rf（独立会话）。"""
    return _build_delivered(sessions_root, reg_csv)


@pytest.fixture
def delivered_fresh(sessions_root, reg_csv, shared_tracking):
    """独立全量会话（function 级）：finish 收局终态化 manifest——不可与只读族共享。"""
    return _build_delivered(sessions_root, reg_csv)


@pytest.fixture
def delivered_rf(sessions_root, reg_csv, shared_tracking):
    """独立单模型终态会话：工具面接缝用例（双模型非必要——降重建成本）。"""
    s = _start(sessions_root, reg_csv)
    sid = s["session_id"]
    ms.apply_action(sid, "select_model", {"family": "RF"})
    ms.apply_action(sid, "set_hp", {"n_estimators": 15})
    ms.apply_action(sid, "run_hpo", {"n_trials": 2})
    ms.apply_action(sid, "final_train_test")
    return s


# ── OOF 图集挂 run ─────────────────────────────────────────────────────────


def test_run_hpo_writes_oof_plot_album(sessions_root, reg_csv, shared_tracking):
    """OOF 图集：run_hpo 产物挂 run 下（run_id 寻址）——图集可按 run 列出。"""
    s = _start(sessions_root, reg_csv)
    sid = s["session_id"]
    ms.apply_action(sid, "select_model", {"family": "RF"})
    ms.apply_action(sid, "set_hp", {"n_estimators": 15})
    ms.apply_action(sid, "run_hpo", {"n_trials": 2})
    m = _manifest(sessions_root, sid)
    oof = m["models"][_K]["oof"]
    plots = (sessions_root / sid / "runs" / oof["run_id"] / "plots").glob("*.png")
    files = sorted(p.name for p in plots)
    assert any("prediction_scatter" in f for f in files)  # 回归图集指向 OOF 源
    assert oof["plots"] and set(oof["plots"]) == set(files)  # manifest 面与盘一致
    oof_df = pd.read_parquet(sessions_root / sid / oof["y_pred"])
    assert len(oof_df) == 96  # 120 行 × 0.8 train（cv=3 → outer test_size 缺省 0.2）
    import matplotlib.pyplot as plt

    assert not plt.get_fignums()  # #92 图集落盘即闭图——600dpi 缓冲 ~80MB/张，泄漏即进程级累积


# ── draw_taylor_diagram（副产物 fan-in + 同一性断言） ───────────────────────


def test_draw_taylor_diagram_multi_model(delivered, sessions_root):
    s = delivered
    m = _manifest(sessions_root, s["session_id"])
    run_ids = [m["models"][k]["run_id"] for k in (_K, _K2)]
    out = draw_taylor_diagram(session_id=s["session_id"], run_ids=run_ids)
    stats = out["stats"]
    assert set(stats) == {_K, _K2}  # stats 键 = combo key（ledger 反查）
    for row in stats.values():
        assert {"correlation", "sigma_ratio", "rmse_total_norm"} <= set(row)  # (corr, σ, rmse)
    assert out["figure"] and str(out["figure"]).endswith(".png")  # 图落盘
    assert Path(out["figure"]).exists()
    import matplotlib.pyplot as plt

    assert not plt.get_fignums()  # #92 泰勒图落盘即闭图（同 OOF 图集纪律）


def test_draw_taylor_diagram_single_model_edge(delivered, sessions_root):
    """单模型：泰勒为多模型对比图——不出图，统计照出（边界显式）。"""
    s = delivered
    m = _manifest(sessions_root, s["session_id"])
    out = draw_taylor_diagram(session_id=s["session_id"], run_ids=[m["models"][_K]["run_id"]])
    assert out["figure"] is None and "单模型" in out["note"]


def test_draw_taylor_diagram_missing_artifact_fail_loud(delivered):
    """run 不在账本终训面 = fail-loud 带可用清单（无终训副产物的 run 不可入图）。"""
    s = delivered
    with pytest.raises(ValueError, match="终训面"):
        draw_taylor_diagram(session_id=s["session_id"], run_ids=["no_such_run"])


def test_taylor_byproducts_refuses_mismatched_y_test():
    """同一性断言（边函数缝）：各模型 y_test 逐位不一致 → 拒绝出图（撒谎的图）。"""
    from mflowy.model.report import taylor_byproducts

    y_test_a = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    y_test_b = pd.DataFrame({"y": [1.0, 2.0, 3.1]})  # 另一 holdout
    y_pred = pd.DataFrame({"y": [1.1, 2.2, 2.9]})
    with pytest.raises(ValueError, match="y_test"):
        taylor_byproducts([("A", y_test_a, y_pred), ("B", y_test_b, y_pred)])
    # 一致的 y_test 正常聚合
    long_df = taylor_byproducts([("A", y_test_a, y_pred), ("B", y_test_a, y_pred)])
    assert set(long_df["model"]) == {"A", "B"} and {"y_name", "y", "y_pred"} <= set(long_df.columns)


# ── modeling_report_skeleton（只读素材面） ──────────────────────────────────


def test_report_skeleton_renders_from_ledger(delivered, sessions_root):
    """skeleton：表格数字与账本逐位相等（抄文件非默写）+ 图片清单 + 叙事槽；
    只读零副作用（manifest 不动）。"""
    sid = delivered["session_id"]
    before = _manifest(sessions_root, sid)
    out = modeling_report_skeleton(session_id=sid)

    entry = before["models"][_K]
    board_row = next(r for r in out["tables"]["board"] if r["key"] == _K)
    assert board_row["u_inner"] == entry["u_inner"] and board_row["u"] == entry["u"]
    assert board_row["rank"] in (1, 2)  # 冻结面名次进表格（rank 不经 π 转写）
    assert board_row["best_params"] == entry["best_params"]  # θ* 转写免 π
    row2 = next(r for r in out["tables"]["board"] if r["key"] == _K2)
    assert row2["u"] == before["models"][_K2]["u"]

    images = {img["key"]: img for img in out["images"]}
    assert set(images) == {_K, _K2}
    for img in images.values():
        assert img["oof_plots"] and img["holdout_byproducts"] == {
            "X_test": "X_test.parquet",
            "y_test": "y_test.parquet",
            "y_pred": "y_pred.parquet",
        }
        assert img["run_id"]
    assert out["narrative_slots"]  # 叙事槽样板（判断归 π）
    assert out["rank_consistency"] is not None  # SH 赌注出账浮出（双冻结模型）
    assert _manifest(sessions_root, sid) == before  # 零副作用


def test_report_skeleton_includes_taylor_after_draw(delivered, sessions_root):
    """收局剧本序（画泰勒 → 看 skeleton）：泰勒图路径进图片清单。"""
    sid = delivered["session_id"]
    m = _manifest(sessions_root, sid)
    out_draw = draw_taylor_diagram(session_id=sid, run_ids=[m["models"][k]["run_id"] for k in (_K, _K2)])
    sk = modeling_report_skeleton(session_id=sid)
    assert sk["taylor"] == out_draw["figure"]


# ── finish(pdf) 单动作 ─────────────────────────────────────────────────────


def test_finish_missing_pdf_fail_loud(delivered, tmp_path):
    """收局带不存在的报告路径 = fail-loud（素材面先经 draw_taylor/skeleton 两工具）。"""
    with pytest.raises(ValueError, match="报告文件不存在"):
        ms.finish_session(delivered["session_id"], pdf=str(_pdf(tmp_path, "nope.pdf").with_name("absent.pdf")))


def test_finish_pdf_required(delivered):
    """BREAKING：pdf 必填——preview 分支退役（无 pdf 调用即 TypeError）。"""
    with pytest.raises(TypeError):
        ms.finish_session(delivered["session_id"])


def test_finish_pdf_adoption_and_verdict(delivered_fresh, sessions_root, tmp_path):
    """真收局：收编 π 产报告（tmp/rename 落 deliverables、内容不验）+ verdict 落账。"""
    sid = delivered_fresh["session_id"]
    pdf = _pdf(tmp_path)
    v = ms.finish_session(sid, pdf=str(pdf))
    verify_verdict(v)
    assert v["accepted"] is True
    assert {k: v["deliverables"][k]["terminal_utility"] for k in (_K, _K2)} == v["terminal_utility"]
    assert v["report"]["file"] == "deliverables/final_report.pdf"
    assert (sessions_root / sid / v["report"]["file"]).read_bytes() == pdf.read_bytes()  # 收编落盘
    assert not list((sessions_root / sid / "deliverables").glob(".*.tmp"))  # tmp 收干净
    m = _manifest(sessions_root, sid)
    assert m["verdict"]["report"]["file"] == "deliverables/final_report.pdf"
    assert any(
        a["action"] == "finish" and "final_report.pdf" in json.dumps(a, ensure_ascii=False) for a in m["decision_log"]
    )

    again = ms.finish_session(sid, pdf=str(pdf))
    assert again["deliverables"] == v["deliverables"]  # 重入幂等
    rewrite = tmp_path / "rewrite.pdf"
    rewrite.write_bytes(b"%PDF-1.4 v2")
    with pytest.raises(ValueError, match="不可变更"):
        ms.finish_session(sid, pdf=str(rewrite))  # 交付报告不可变更（审计面）


def test_modeling_finish_tool_face(delivered_rf, tmp_path):
    """接缝（session→tools）：modeling_finish 的 pdf 透传可达（工具面=会话面）。"""
    sid = delivered_rf["session_id"]
    pdf = _pdf(tmp_path)
    v = modeling_finish(session_id=sid, pdf=str(pdf))
    verify_verdict(v)
    assert v["accepted"] is True and v["report"]["file"].startswith("deliverables/")


def test_taylor_figure_name_single_source():
    """#83 单源红线：``taylor_diagram.png`` 字面量全仓包源码唯一（常量定义处）——skeleton 清单与图写入方同源。"""
    import mflowy.model.report

    pkg = Path(mflowy.model.report.__file__).parent
    root = pkg.parents[4]  # 仓根（editable 安装即工作树源码）
    occurrences = [
        f"{p.relative_to(root)}:{i}"
        for p in sorted(root.joinpath("packages").rglob("*.py"))
        if "tests" not in p.parts
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "taylor_diagram.png" in line
    ]
    assert occurrences == ["packages/builtin_plugins/model/mflowy/model/report.py:27"], occurrences


def test_delivered_fixture_is_module_scoped():
    """#92 回归钉：delivered 模块级共享——scope 降级即 9 用例 × 4 管线线性重建，
    单文件峰值 10GB（全量 ~20GB，取证见 #92）。锁 scope 不锁行为。"""
    assert delivered._fixture_function_marker.scope == "module"
