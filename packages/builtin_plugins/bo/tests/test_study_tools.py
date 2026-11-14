"""六工具 + study 状态载体集成契约（#180/#181/#182/#183 裁定钉测）。

真 sqlite study（无 mock——optuna 行为面即契约对象）：开局三形态/20 门/ask 铺列/
在途避让/回填五形态/信号批末/导出/迁移式重建，fail-loud 面逐条钉。
"""

import numpy as np
import pandas as pd
import pytest
from mflowy.bo import tools

SPACE = {
    "temperature": {"kind": "continuous", "low": 20.0, "high": 90.0, "dtype": "float"},
    "enzyme": {"kind": "discrete", "choices": ["WT", "E1"]},
}
MATCH_SPACE = dict(SPACE)


@pytest.fixture
def history(tmp_path) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "temperature": rng.uniform(20, 90, 25).round(2),
            "ph": rng.uniform(5, 8, 25).round(2),
            "enzyme": rng.choice(["WT", "E1"], 25),
            "yield": rng.uniform(10, 50, 25).round(3),
        }
    )
    df.to_parquet(tmp_path / "wet_history.parquet")
    return df


@pytest.fixture
def study(tmp_path, history) -> dict:
    out = tools.bo_load_or_init(
        study_name="s1",
        model_run_id="run-A",
        data=str(tmp_path / "wet_history.parquet"),
        target_column="yield",
        direction="maximize",
        search_space=SPACE,
        seed=7,
    )
    assert out["action"] == "init" and out["n_history"] == 25
    return out


# ── 开局：init / 20 门 / 撞名 / 双面拦截 ───────────────────────────────────────


def test_load_or_init_requires_data_or_db(tmp_path):
    with pytest.raises(ValueError, match="至少一个"):
        tools.bo_load_or_init(study_name="s1", model_run_id="r")


def test_init_min_history_gate(tmp_path, history):
    short = history.head(10)
    short.to_parquet(tmp_path / "short.parquet")
    with pytest.raises(ValueError, match="开局最小历史量 20"):
        tools.bo_load_or_init(
            study_name="s2", model_run_id="r", data=str(tmp_path / "short.parquet"),
            target_column="yield", direction="maximize", search_space=SPACE,
        )


def test_init_name_collision_fails_loud(tmp_path, study):
    with pytest.raises(FileExistsError, match="撞名"):
        tools.bo_load_or_init(
            study_name="s1", model_run_id="r", data=study_study_parquet(tmp_path),
            target_column="yield", direction="maximize", search_space=SPACE,
        )


def study_study_parquet(tmp_path) -> str:
    return str(tmp_path / "wet_history.parquet")


def test_init_requires_space_direction_target(tmp_path, history):
    with pytest.raises(ValueError, match="必填"):
        tools.bo_load_or_init(study_name="s3", model_run_id="r", data=study_study_parquet(tmp_path))
    with pytest.raises(ValueError, match="schema 同构"):
        tools.bo_load_or_init(
            study_name="s3", model_run_id="r", data=study_study_parquet(tmp_path),
            target_column="yield", direction="maximize",
            search_space={"pressure": {"kind": "continuous", "low": 0, "high": 5}},
        )


def test_match_init_requires_band(tmp_path, history):
    with pytest.raises(ValueError, match="需要 band"):
        tools.bo_load_or_init(
            study_name="s4", model_run_id="r", data=study_study_parquet(tmp_path),
            target_column="yield", direction="match", search_space=SPACE,
        )
    out = tools.bo_load_or_init(
        study_name="s4", model_run_id="r", data=study_study_parquet(tmp_path),
        target_column="yield", direction="match", search_space=MATCH_SPACE, band=[40, 60],
    )
    assert out["action"] == "init"


# ── 推荐：ask 铺列 + 在途避让 ──────────────────────────────────────────────────


def test_ask_full_schema_csv_and_running_trials(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=4)
    assert ask["n_asked"] == 4
    df = pd.read_csv(ask["candidates_csv"])
    assert list(df.columns) == ["trial", "temperature", "ph", "enzyme"]  # 非搜索维空列（#179 R1）
    assert df["ph"].isna().all() and df["temperature"].notna().all()
    numbers = [t["trial"] for t in ask["trials"]]
    assert numbers == sorted(numbers) and numbers[0] >= 25  # trial number 直用（历史 25 条占 0-24）

    # 在途避让：同一批重复 ask 不与既有 RUNNING 撞点
    ask2 = tools.bo_ask(study["study_db"], n_candidates=4)
    keys1 = {tuple(sorted(t["params"].items())) for t in ask["trials"]}
    keys2 = {tuple(sorted(t["params"].items())) for t in ask2["trials"]}
    assert not (keys1 & keys2)


# ── 剪枝 / 多样性落选 ──────────────────────────────────────────────────────────


def test_prune_states_and_reasons(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=3)
    n = [t["trial"] for t in ask["trials"]]
    out = tools.bo_prune(study["study_db"], trials=n[:2], reason="out_of_band")
    assert out["pruned"] == n[:2]
    with pytest.raises(ValueError, match="reason 须为"):
        tools.bo_prune(study["study_db"], trials=[n[2]], reason="whatever")
    with pytest.raises(ValueError, match="已终态"):  # 先到为准，无覆盖入口
        tools.bo_prune(study["study_db"], trials=[n[0]], reason="user_rejected")


def test_diversity_check_marks_rejected_fail(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=5)
    numbers = [t["trial"] for t in ask["trials"]]
    out = tools.bo_diversity_check(study["study_db"], candidates=numbers, batch_size=2)
    assert len(out["selected"]) == 2 and len(out["rejected"]) == 3
    assert set(out["selected"]) | set(out["rejected"]) == set(numbers)
    with pytest.raises(ValueError, match="已终态"):
        tools.bo_prune(study["study_db"], trials=out["rejected"][:1], reason="user_rejected")


# ── 回填：五形态 + fail-loud ──────────────────────────────────────────────────


def test_tell_done_and_appended_replicate(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=2)
    head = ask["trials"][0]
    out = tools.bo_tell(
        study["study_db"],
        [
            {"trial": head["trial"], "status": "done", "measured_value": 42.0, "label": "close",
             "round": 1, "conditions": {"ph": 6.5}},
            {"params": head["params"], "measured_value": 42.3, "source": "recommended",
             "replicate_of": head["trial"], "label": "close"},  # 平行样余条
        ],
    )
    assert [t["state"] for t in out["told"]] == ["COMPLETE", "COMPLETE"]
    assert "signal" not in out or out["signal"]["state"]["positive"] >= 1  # 未触发静默


def test_tell_failed_attribution_map(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=3)
    n = [t["trial"] for t in ask["trials"]]
    out = tools.bo_tell(
        study["study_db"],
        [
            {"trial": n[0], "status": "failed", "attribution": "conditional"},
            {"trial": n[1], "status": "failed", "attribution": "operational"},
            {"trial": n[2], "status": "abandoned"},
        ],
    )
    states = {t["trial"]: t["state"] for t in out["told"]}
    assert states == {n[0]: "PRUNED", n[1]: "FAIL", n[2]: "FAIL"}
    ask2 = tools.bo_ask(study["study_db"], n_candidates=1)
    with pytest.raises(ValueError, match="attribution 必填"):
        tools.bo_tell(study["study_db"], [{"trial": ask2["trials"][0]["trial"], "status": "failed"}])
    with pytest.raises(ValueError, match="不存在"):
        tools.bo_tell(study["study_db"], [{"trial": 999, "status": "done", "measured_value": 1.0}])


def test_tell_deviated_variant_semantics(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=2)
    base = ask["trials"][0]
    measured = {**base["params"], "temperature": 55.5}
    out = tools.bo_tell(
        study["study_db"],
        [{"trial": base["trial"], "status": "deviated", "params": measured,
          "measured_value": 31.0, "variant": True, "label": "close", "conditions": {"ph": 7.0}}],
    )
    row = out["told"][0]
    assert row["original"] == "FAIL(variant_replaced)" and row["source"] == "recommended"

    base2 = ask["trials"][1]
    out2 = tools.bo_tell(
        study["study_db"],
        [{"trial": base2["trial"], "status": "deviated", "params": {**base2["params"], "temperature": 60.0},
          "measured_value": 30.0, "variant": False}],
    )
    assert "RUNNING" in out2["told"][0]["original"]  # 默认自发追加——原推荐留在途


def test_tell_fail_loud_forms(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=2)
    n = [t["trial"] for t in ask["trials"]]
    with pytest.raises(ValueError, match="二选一"):
        tools.bo_tell(study["study_db"], [{"trial": n[0], "params": {"temperature": 50, "enzyme": "WT"}, "status": "done"}])
    with pytest.raises(ValueError, match="未知 status"):
        tools.bo_tell(study["study_db"], [{"trial": n[0], "status": "weird"}])
    with pytest.raises(ValueError, match="schema 外列"):
        tools.bo_tell(study["study_db"], [{"trial": n[0], "status": "done", "measured_value": 1.0,
                                           "conditions": {"ghost_col": 1}}])
    with pytest.raises(ValueError, match="超出搜索分布"):
        tools.bo_tell(study["study_db"], [{"params": {"temperature": 500.0, "enzyme": "WT"}, "measured_value": 1.0}])
    tools.bo_tell(study["study_db"], [{"trial": n[0], "status": "done", "measured_value": 5.0}])
    with pytest.raises(ValueError, match="已终态"):  # 两路先到为准（#181 R3）
        tools.bo_tell(study["study_db"], [{"trial": n[0], "status": "done", "measured_value": 6.0}])


def test_tell_match_band_required(tmp_path, history):
    out = tools.bo_load_or_init(
        study_name="m1", model_run_id="r", data=str(tmp_path / "wet_history.parquet"),
        target_column="yield", direction="match", search_space=MATCH_SPACE, band=[40, 60],
    )
    ask = tools.bo_ask(out["study_db"], n_candidates=1)
    with pytest.raises(ValueError, match="回填需 band"):
        tools.bo_tell(out["study_db"], [{"trial": ask["trials"][0]["trial"], "status": "done", "measured_value": 41.0}])
    # match 编码：带外实测值 → hinge 距离入 trial.value，raw 恒存 measured_value attr
    told = tools.bo_tell(out["study_db"], [
        {"trial": ask["trials"][0]["trial"], "status": "done", "measured_value": 80.0, "label": "not_counted"},
    ], band=[40, 60])
    assert told["told"][0]["state"] == "COMPLETE"


def test_tell_window_gate_pair_validation(study):
    with pytest.raises(ValueError, match="成对"):
        tools.bo_tell(study["study_db"], [], window_k=5)
    with pytest.raises(ValueError, match="冗余"):
        tools.bo_tell(study["study_db"], [], window_k=5, window_m=3, negative_gate=3)


# ── 学习信号：计数制 + 纪元归档重置（#177 R4/#182 R3）─────────────────────────


def test_signal_positive_gate_and_silence(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=1)
    out = tools.bo_tell(
        study["study_db"],
        [{"trial": ask["trials"][0]["trial"], "status": "done", "measured_value": 42.0, "label": "close"}],
        positive_gate=1,
    )
    assert "signal" in out and out["signal"]["state"]["positive_triggered"]  # 触发返回门禁与建议
    assert "继续推荐" in out["signal"]["suggestion"]

    ask2 = tools.bo_ask(study["study_db"], n_candidates=1)
    quiet = tools.bo_tell(
        study["study_db"],
        [{"trial": ask2["trials"][0]["trial"], "status": "done", "measured_value": 42.0, "label": "close"}],
        positive_gate=100,  # 未触发静默
    )
    assert "signal" not in quiet


def test_signal_run_id_rebind_archives_epoch(study):
    ask = tools.bo_ask(study["study_db"], n_candidates=1)
    tools.bo_tell(
        study["study_db"],
        [{"trial": ask["trials"][0]["trial"], "status": "done", "measured_value": 42.0, "label": "close"}],
        positive_gate=100,
    )
    re = tools.bo_load_or_init(study_name="s1", model_run_id="run-B", db=study["study_db"])
    assert re["action"] == "continue"
    assert re["run_id_rebind"]["archived_run_id"] == "run-A"
    assert re["run_id_rebind"]["reset"]["positive"] == 0  # 清零一气呵成
    # 重绑已归档 run_id（回滚旧模型）= 覆盖该键归档、重新计数
    re2 = tools.bo_load_or_init(study_name="s1", model_run_id="run-A", db=study["study_db"])
    assert re2["run_id_rebind"]["archived_run_id"] == "run-B"


# ── 导出（#182 R1）────────────────────────────────────────────────────────────


def test_export_complete_only_with_bookkeeping(study, tmp_path):
    ask = tools.bo_ask(study["study_db"], n_candidates=3)
    n = [t["trial"] for t in ask["trials"]]
    tools.bo_prune(study["study_db"], trials=[n[0]], reason="hard_rule_violation")
    tools.bo_tell(
        study["study_db"],
        [{"trial": n[1], "status": "done", "measured_value": 42.0, "label": "close", "round": 2,
          "conditions": {"ph": 6.5}}],
    )
    out = tools.bo_export_dataset(study["study_db"], str(tmp_path / "ds.parquet"))
    df = pd.read_parquet(out["output_path"])
    assert out["n_rows"] == 26  # 25 历史 + 1 回填（PRUNED 不进）
    assert list(df.columns) == ["temperature", "ph", "enzyme", "measured_value", "round", "replicate_of", "source"]
    row = df[df["round"] == 2].iloc[0]
    assert row["ph"] == 6.5 and row["measured_value"] == 42.0  # conditions 钉值 + raw 实测


# ── 迁移式重建（#180 R4）──────────────────────────────────────────────────────


def test_rebuild_bounds_change_migrates_terminal_states(study, tmp_path):
    ask = tools.bo_ask(study["study_db"], n_candidates=3)
    n = [t["trial"] for t in ask["trials"]]
    tools.bo_prune(study["study_db"], trials=[n[0]], reason="out_of_band")
    tools.bo_tell(study["study_db"], [{"trial": n[1], "status": "done", "measured_value": 42.0, "label": "close"}])
    # n[2] 留 RUNNING → 重建终局化 FAIL

    narrowed = dict(SPACE)
    narrowed["temperature"] = {"kind": "continuous", "low": 15.0, "high": 95.0, "dtype": "float"}  # 物理边界变（放宽——收窄越界即 fail-loud，见下例）
    out = tools.bo_load_or_init(
        study_name="rebuilt", model_run_id="run-A", db=study["study_db"],
        direction="maximize", search_space=narrowed, seed=7,
    )
    assert out["action"] == "rebuild"
    report = out["migration_report"]
    assert report["finalized_running"] == 1  # RUNNING → FAIL（rebuild_finalized，落源 db）
    assert report["migrated"]["complete"] >= 26 and report["migrated"]["pruned_fail"] >= 1  # 同空间重建：负面判断照迁


def test_rebuild_out_of_bounds_fails_loud(study, tmp_path, history):
    # 物理边界收窄到既有数据之外（历史 20-90 采样 → 收窄到 21 内必有越界）→ fail-loud 列名
    narrowed = dict(SPACE)
    narrowed["temperature"] = {"kind": "continuous", "low": 20.0, "high": 21.0, "dtype": "float"}
    with pytest.raises(ValueError, match="temperature"):
        tools.bo_load_or_init(
            study_name="rebuilt2", model_run_id="run-A", db=study["study_db"],
            direction="maximize", search_space=narrowed, seed=7,
        )


def test_rebuild_new_dim_drops_negative_and_backfills(study, tmp_path, history):
    ask = tools.bo_ask(study["study_db"], n_candidates=3)
    n = [t["trial"] for t in ask["trials"]]
    tools.bo_prune(study["study_db"], trials=[n[0]], reason="hard_rule_violation")
    tools.bo_tell(
        study["study_db"],
        [{"trial": n[1], "status": "done", "measured_value": 42.0, "label": "close",
          "conditions": {"ph": 6.9}}],
    )
    wider = dict(SPACE)
    wider["ph"] = {"kind": "continuous", "low": 5.0, "high": 8.0, "dtype": "float"}  # 新增搜索维
    out = tools.bo_load_or_init(
        study_name="rebuilt3", model_run_id="run-A", db=study["study_db"],
        direction="maximize", search_space=wider, seed=7,
        data=str(tmp_path / "wet_history.parquet"),  # 湿历史补新增维值
    )
    report = out["migration_report"]
    assert report["backfilled_dims"] == ["ph"]
    assert report["dropped"]["pruned_fail"] == 1  # 新增维度——旧空间负面判断不可比不迁
    exp = tools.bo_export_dataset(out["study_db"], str(tmp_path / "r3.parquet"))
    assert "ph" in exp["columns"]


def test_continue_rejects_ambiguous_anchors(study, tmp_path, history):
    with pytest.raises(ValueError, match="不收 data"):
        tools.bo_load_or_init(
            study_name="s1", model_run_id="run-A", db=study["study_db"],
            data=str(tmp_path / "wet_history.parquet"),
        )  # 无 search_space + data 同传 = 意图不明
    with pytest.raises(ValueError, match="须一并传入"):
        tools.bo_load_or_init(
            study_name="s1", model_run_id="run-A", db=study["study_db"], search_space=SPACE,
        )  # 比对指纹需要 direction
