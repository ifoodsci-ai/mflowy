"""bo-bench 场景 overlay 与回合 harness（#193 题 1/3 → #231）。

- 场景 S1 规则贴边 / S2 带窗（match + band，hi < 全局峰）/ S3 漂移（第 8 轮真值
  换形，f_pre 训于旧 regime）——overlay 不改函数族本体
- harness 消费**真实工具面**（bo_load_or_init/bo_ask/bo_prune/bo_diversity_check/
  bo_tell——#193 单源红线：不得旁路自研平行循环）；agent 侧判断（剪枝/落诊标签）
  在 harness 内确定性复刻（读卡规则的函数化）
- 度量 m1–m7（#193 题 3 协议表）：轨迹/带命中/可行产出/触发时延/漂移浪费/批重复
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .fpre import SurrogateFPre
from .functions import FAMILIES, Family

# 协议常量（#193 题 1 协议表——主门口径）
WARMUP_H = 20
ROUNDS_R = 15
BATCH_B = 6
OVERSAMPLE_K = 5  # n_candidates = k × B = 30
DRIFT_ROUND = 8  # S3：第 8 轮起真值换形
MEAS_NOISE = 0.01  # 种子化测量噪声（相对量级）
FPRE_SAMPLES = 240  # 代理训练样本数（域内拉丁超立方）
POSITIVE_GATE = 100  # 卡缺省（#178）——断言语义不断言具体数
NEGATIVE_GATE = 3


def _lhs_sample(dims: dict, n: int, seed: int) -> list[dict[str, Any]]:
    """域内拉丁超立方（共同随机数：同 seed 跨臂同历史——隔离采样器变量）。"""
    rng = np.random.default_rng(seed)
    cont = {c: np.sort(rng.uniform(0, 1, n)) for c in dims}
    for c in cont:
        cont[c] = rng.permutation(cont[c])  # 维间解耦（分层后置换）
    rows: list[dict] = []
    for i in range(n):
        row: dict[str, Any] = {}
        for c, spec in dims.items():
            if spec["kind"] == "continuous":
                row[c] = float(spec["low"] + cont[c][i] * (spec["high"] - spec["low"]))
            else:
                row[c] = spec["choices"][int(rng.integers(len(spec["choices"])))]
        rows.append(row)
    return rows


class Scenario:
    """场景 overlay：方向/带/硬规则/漂移——包住族不改本体。"""

    def __init__(self, name: str, family: Family) -> None:
        self.name = name
        self.family = family
        self.direction: str = family.direction
        self.band: tuple[float, float] | None = None
        self.hard_rule: str | None = None  # S1：连续维上限谓词（贴边约束）
        self.drifts = False  # S3
        self.drift_shift = 8.0  # 恒幅抬升缺省（make_scenario 按 fpre 半宽锚定覆盖）

    def true_f(self, params: dict[str, Any], round_no: int) -> float:
        f = self.family.true_f(params)
        if self.drifts and round_no >= DRIFT_ROUND:
            f = f + self.drift_shift  # 换形：恒幅抬升（幅度锚代理半宽——m5/m6 必可达）
        if self.hard_rule is not None:
            cap = self._rule_cap
            assert self.hard_rule is not None and cap is not None  # 收窄（_cap_setup 已置）
            if float(params[self.hard_rule]) > cap:
                f = f + 10.0 * (float(params[self.hard_rule]) - cap)  # 越界惩罚（硬规则的可观测面）
        return f

    def _cap_setup(self) -> None:
        """S1：把约束上限设在（近似）最优的 0.95 分位处——最优贴边。"""
        assert self.hard_rule is not None  # 调用序保证（make_scenario 先置 hard_rule）
        rule = self.hard_rule
        spec = self.family.dims[rule]
        grid = np.linspace(spec["low"], spec["high"], 50)
        best_x, best_v = 0.0, float("inf")
        base = _lhs_sample(self.family.dims, 64, seed=777)
        for x in grid:
            v = min(self.family.true_f({**row, rule: float(x)}) for row in base)
            if v < best_v:
                best_x, best_v = float(x), v
        self._rule_cap: float = best_x * 0.92 + spec["low"] * 0.08  # 贴边（最优恰在约束内缘）

    def violates_rule(self, params: dict) -> bool:
        return self.hard_rule is not None and float(params[self.hard_rule]) > self._rule_cap


def make_scenario(scenario: str, family_name: str) -> Scenario:
    sc = Scenario(scenario, FAMILIES[family_name])
    if scenario == "S1":
        sc.hard_rule = "x1" if "x1" in family_dims(sc.family) else sorted(cont_dims(sc.family))[0]
        sc._cap_setup()
    elif scenario == "S2":
        sc.direction = "match"
        f = sc.family
        samples = [f.true_f(r) for r in _lhs_sample(f.dims, 400, seed=4242)]
        lo, hi = np.quantile(samples, [0.02, 0.10])  # 近最优窄带（随机命中难、BO 应聚集）
        sc.band = (float(lo), float(hi))
    elif scenario == "S3":
        sc.drifts = True  # drift_shift 由 run_bench_arm 按 fpre 半宽锚定（见 fpre 拟合后）
    elif scenario != "base":
        raise ValueError(f"未知场景 {scenario!r}（可用: base/S1/S2/S3）")
    return sc


def cont_dims(family: Family) -> list[str]:
    return [c for c, s in family.dims.items() if s["kind"] == "continuous"]


def family_dims(family: Family) -> list[str]:
    return list(family.dims)


def run_bench_arm(
    *,
    scenario: str,
    family_name: str,
    seed: int,
    arm: str,
    workdir: Path,
    warmup_h: int = WARMUP_H,
    rounds: int = ROUNDS_R,
) -> dict:
    """跑一臂（family × scenario × seed × arm）→ 度量记录（m1–m7 素材）。

    arm ∈ A1（tpe mv+cl 生产臂）/ A2（random）/ A3（tpe univariate——仅交互族有义）。
    工具面消费恒真实；arm 差异只在 study engine 的 sampler 换装（study.SAMPLER_KINDS）。
    """
    from mflowy.bo import study as bo_study
    from mflowy.bo import tools as bo_tools

    sc = make_scenario(scenario, family_name)
    family = sc.family
    rng = np.random.default_rng(seed)
    study_dir = workdir / f"{family_name}_{scenario}_s{seed}_{arm}"
    study_dir.mkdir(parents=True, exist_ok=True)

    # ── f_pre 训练样本（旧 regime——S3 漂移后 f_pre 不更新）──────────────────
    fpre_rows = _lhs_sample(family.dims, FPRE_SAMPLES, seed=seed + 9000)
    fpre = SurrogateFPre().fit(pd.DataFrame(fpre_rows), np.array([family.true_f(r) for r in fpre_rows]))

    # ── warm-up 历史（共同随机数：同 family×seed 跨臂同历史）────────────────
    history = _lhs_sample(family.dims, warmup_h, seed=seed + 31337)
    hist_df = pd.DataFrame(history)
    hist_df["y"] = [family.true_f(r) for r in history]
    hist_path = study_dir / "warmup.parquet"
    hist_df.to_parquet(hist_path, index=False)

    if sc.drifts:
        sc.drift_shift = 12.0 * fpre.half_width  # 幅度 = 12×半宽（5×hw 判定界恒分辨——m5 触发必可达）

    init = bo_tools.bo_load_or_init(
        study_name=f"bobench-{family_name}-{scenario}-s{seed}-{arm}",
        model_run_id=f"bo-bench-{arm}",
        data=str(hist_path),
        target_column="y",
        direction=sc.direction,
        search_space=family.dims,
        band=list(sc.band) if sc.band else None,
        seed=seed,
    )
    study_db = init["study_db"]
    if arm != "A1":  # 臂换装：engine.sampler patch（study.SAMPLER_KINDS 词表内）
        st = bo_study.open_study(Path(study_db))
        engine = dict(st.user_attrs[bo_study.ATTR_ENGINE])
        engine["sampler"] = "random" if arm == "A2" else "tpe_univariate"
        st.set_user_attr(bo_study.ATTR_ENGINE, engine)

    # ── 回合循环（真实工具面）────────────────────────────────────────────────
    best_seen = float(hist_df["y"].min())
    wet_values: list[float] = []
    band_hits = 0
    feasible_kept = 0
    infeasible_total = 0
    negative_trigger_round: int | None = None
    wasted_after_drift = 0
    seen_param_keys: set[tuple] = set()
    batch_duplicates = 0

    for round_no in range(1, rounds + 1):
        ask = bo_tools.bo_ask(study_db, BATCH_B * OVERSAMPLE_K)
        trials = [t["trial"] for t in ask["trials"]]
        params_list = [t["params"] for t in ask["trials"]]

        # agent 侧剪枝（确定性读卡规则）：硬规则违例 / 预测区间整段出带
        kept, pruned_rule, pruned_band = [], [], []
        for n, p in zip(trials, params_list, strict=True):
            key = tuple(sorted((k, str(v)) for k, v in p.items()))
            if key in seen_param_keys:
                batch_duplicates += 1  # m7（理论上 constant_liar 臂内 = 0）
            if sc.violates_rule(p):
                pruned_rule.append(n)
            else:
                point, hw = fpre.predict(p)
                if sc.band is not None and (point + hw < sc.band[0] or point - hw > sc.band[1]):
                    pruned_band.append(n)
                else:
                    kept.append(n)
        if pruned_rule:
            bo_tools.bo_prune(study_db, pruned_rule, "hard_rule_violation")
        if pruned_band:
            bo_tools.bo_prune(study_db, pruned_band, "out_of_band")
        infeasible_total += len(pruned_rule)
        feasible_kept += len(kept)

        # 多样性收批（kept > B 时）
        if len(kept) > BATCH_B:
            div = bo_tools.bo_diversity_check(study_db, kept, batch_size=BATCH_B)
            selected = div["selected"]
        else:
            selected = kept

        # 湿实验（真值求值 + 种子化噪声）→ 回填 + 信号批末附带
        rows, round_values = [], []
        sel_params = dict(zip(trials, params_list, strict=True))
        for n in selected:
            p = sel_params[n]
            y = sc.true_f(p, round_no)
            y_noisy = y * (1.0 + MEAS_NOISE * rng.normal())
            point, hw = fpre.predict(p)
            label = (
                "close" if abs(y_noisy - point) <= 5 * hw else "severe_mismatch"
            )  # 5×hw：LHS 面 0 尾 + TPE 聚集面余量
            rows.append({"trial": n, "status": "done", "measured_value": y_noisy, "label": label, "round": round_no})
            round_values.append(y_noisy)
            seen_param_keys.add(tuple(sorted((k, str(v)) for k, v in p.items())))
        if rows:
            told = bo_tools.bo_tell(
                study_db,
                rows,
                band=list(sc.band) if sc.band else None,
                positive_gate=POSITIVE_GATE,
                negative_gate=NEGATIVE_GATE,
            )
            sig = (told.get("signal") or {}).get("state") or {}  # 信封二联：触发才在场（未触发静默 #180 R1）
            if negative_trigger_round is None and sig.get("negative_triggered"):
                negative_trigger_round = round_no
        if round_values:
            wet_values.extend(round_values)
            best_seen = min(best_seen, min(round_values))
            if sc.band is not None:
                band_hits += sum(1 for v in round_values if sc.band[0] <= v <= sc.band[1])
        if sc.drifts and round_no >= DRIFT_ROUND and negative_trigger_round is None:
            wasted_after_drift += len(rows)  # m6：触发前漂移期无效推送

    return {
        "family": family_name,
        "scenario": scenario,
        "seed": seed,
        "arm": arm,
        "direction": sc.direction,
        "band": list(sc.band) if sc.band else None,
        "known_optimum": family.known_optimum,
        "wet_values": wet_values,
        "cumulative_best_trajectory": list(np.minimum.accumulate(wet_values)) if wet_values else [],
        "final_best": best_seen,
        "band_hits": band_hits,
        "feasible_rate": feasible_kept / max(1, feasible_kept + infeasible_total),
        "negative_trigger_round": negative_trigger_round,
        "drift_trigger_delay": (negative_trigger_round - DRIFT_ROUND if sc.drifts and negative_trigger_round else None),
        "wasted_after_drift": wasted_after_drift if sc.drifts else None,
        "batch_duplicates": batch_duplicates,
    }
