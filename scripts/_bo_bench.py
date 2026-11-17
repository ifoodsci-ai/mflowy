"""bo-bench 矩阵 runner（#193 题 3 通过门 → 执行票 #231）：6 族 × 场景 × 臂 × 种子。

人肉门（发布前跑，仿 e2e-bench 纪律——不进 CI）：

    uv run python scripts/_bo_bench.py --seeds 25 --out .bench/bo        # 主门全矩阵
    uv run python scripts/_bo_bench.py --seeds 8 --families rosenbrock4  # 开发环

产出（``--out`` 目录）：

- ``runs/<family>_<scenario>_s<seed>_<arm>.json``——每臂度量记录（m1–m7 素材）
- ``traj_<family>_<scenario>.png``——累计最优轨迹（25 种子中位数 + IQR 带，
  OptunaHub ``plot_target_over_time`` 同型——#82 防护③ 收敛证据形态）
- ``report.md`` + ``gates.json``——G1–G5 判定 + 门数值（首跑标定，越限自证走
  ``BENCH_FLOOR_BUMP=新值`` 环境变量同 e2e-bench 纪律）

臂矩阵（#193 裁定裁剪——预算 ≈1h 串行 @25 种子）：A1 全 6 族；A2 全 6 族；
A3 仅 3 交互族；场景 × 族默认全 base + 各场景专测族（S1/S2@Hartmann6，S3@Ackley8）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# 协议矩阵（#193 题 3 裁定）
INTERACTION_FAMILIES = ("rosenbrock4", "morphed_hartmann", "denaturation_cliff")  # A3 裁剪 + G2 门
SCENARIO_FAMILIES = {  # 场景专测族（base 全族恒跑）
    "base": None,  # 全 6 族
    "S1": ("hartmann6",),
    "S2": ("hartmann6",),
    "S3": ("ackley8",),
}

# 门初值（#193 临别值——首跑标定后冻结进 CALIBRATED_GATES，越限自证 BENCH_FLOOR_BUMP）
PROVISIONAL_GATES = {
    "G1_min_superior_families": 4,  # A1 vs A2 配对单侧 Wilcoxon p<0.05 显著优 ≥4/6
    "G1_max_inferior_families": 0,  # 显著劣 = 0
    "G1_min_median_superior": 5,  # 中位数优 ≥5/6
    "G2_min_superior_interaction": 2,  # 3 交互族 A1 vs A3 显著优 ≥2
    "G2_max_inferior": 0,  # 全 6 族显著劣 = 0
    "G3_band_ratio": 1.3,  # S2 带命中中位数 ≥ 随机臂 1.3 倍
    "G4_max_trigger_delay_rounds": 2,  # 漂移后 ≤2 轮触发（≥95% 种子）
    "G4_max_seed_share_missing": 0.05,
    "G4_max_wasted_multiple_of_B": 1.5,  # 触发前无效湿点 ≤ 1.5×B
    "G5_max_batch_duplicates": 0,  # 批内精确重复 = 0（constant_liar 自证）
    "G5_min_feasible_rate": 0.80,  # S1 可行候选产出率下限（首跑标定冻结）
}
try:  # 越限自证（e2e-bench BENCH_FLOOR_BUMP 同纪律）
    if bump := os.environ.get("BENCH_FLOOR_BUMP"):
        PROVISIONAL_GATES.update(json.loads(bump))
except json.JSONDecodeError as e:  # pragma: no cover
    raise SystemExit(f"BENCH_FLOOR_BUMP 须为 JSON dict——{e}") from e


def paired_wilcoxon_superiority(a: list[float], b: list[float]) -> tuple[bool, bool]:
    """配对单侧 Wilcoxon（a 优 b？a 劣 b？）——minimize 域值小为优。"""
    from scipy.stats import wilcoxon

    diff = np.array(a) - np.array(b)
    if np.allclose(diff, 0):
        return False, False
    try:
        p_superior = wilcoxon(diff, alternative="less").pvalue  # a 显著小
        p_inferior = wilcoxon(diff, alternative="greater").pvalue
    except ValueError:  # 全零差等退化面归「无显著性」
        return False, False
    return bool(p_superior < 0.05), bool(p_inferior < 0.05)


def run_matrix(seeds: int, families: list[str] | None, out: Path) -> dict:
    from mflowy.bo.bench import FAMILIES, run_bench_arm

    all_families = list(FAMILIES)
    if families:
        unknown = set(families) - set(all_families)
        if unknown:
            raise SystemExit(f"未知族 {sorted(unknown)}（可用: {all_families}）")
        all_families = families

    plan: list[tuple[str, str, str]] = []  # (family, scenario, arm)
    for f in all_families:
        plan.append((f, "base", "A1"))
        plan.append((f, "base", "A2"))
        if f in INTERACTION_FAMILIES:
            plan.append((f, "base", "A3"))
    for sc, fams in SCENARIO_FAMILIES.items():
        if fams:
            for f in fams:
                if f in all_families:
                    plan.append((f, sc, "A1"))
                    plan.append((f, sc, "A2"))

    runs: dict[tuple[str, str, str, int], dict] = {}
    with tempfile.TemporaryDirectory(prefix="bobench-") as tmp:
        workdir = Path(tmp)
        done = 0
        for family, scenario, arm in plan:
            for seed in range(1, seeds + 1):
                rec = run_bench_arm(scenario=scenario, family_name=family, seed=seed, arm=arm, workdir=workdir)
                runs[(family, scenario, arm, seed)] = rec
                done += 1
                print(
                    f"[{done}/{len(plan) * seeds}] {family}/{scenario}/{arm}/s{seed} "
                    f"best={rec['final_best']:.4g} dup={rec['batch_duplicates']}",
                    file=sys.stderr,
                )

    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    runs_dir.mkdir(exist_ok=True)
    for (family, scenario, arm, seed), rec in runs.items():
        (runs_dir / f"{family}_{scenario}_s{seed}_{arm}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    return runs


def plot_trajectories(runs: dict, out: Path, seeds: int) -> None:
    """收敛证据（#82 防护③）：每（族×场景×臂）累计最优轨迹——种子中位数 + IQR 带。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted({(f, sc, arm) for f, sc, arm, _ in runs})
    for family, scenario, arm in keys:
        trajs = [runs[(family, scenario, arm, s)]["cumulative_best_trajectory"] for s in range(1, seeds + 1)]
        trajs = [t for t in trajs if t]
        if not trajs:
            continue
        n = min(len(t) for t in trajs)
        mat = np.array([t[:n] for t in trajs])
        med, q1, q3 = np.median(mat, axis=0), np.quantile(mat, 0.25, axis=0), np.quantile(mat, 0.75, axis=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, n + 1), med, label=f"{arm} 中位数")
        ax.fill_between(range(1, n + 1), q1, q3, alpha=0.25, label="IQR")
        ax.set_xlabel("湿点数")
        ax.set_ylabel("累计最优湿观测")
        ax.set_title(f"{family}/{scenario}（{len(trajs)} 种子）")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"traj_{family}_{scenario}_{arm}.png", dpi=150)
        plt.close(fig)


def evaluate_gates(runs: dict, seeds: int) -> dict:
    from mflowy.bo.bench import FAMILIES

    base_families = [f for f in FAMILIES if (f, "base", "A1", 1) in runs]
    g1: dict[str, Any] = {"superior": [], "inferior": [], "median_superior": 0}
    for f in base_families:
        a1 = [runs[(f, "base", "A1", s)]["final_best"] for s in range(1, seeds + 1)]
        a2 = [runs[(f, "base", "A2", s)]["final_best"] for s in range(1, seeds + 1)]
        sup, inf = paired_wilcoxon_superiority(a1, a2)
        if sup:
            g1["superior"].append(f)
        if inf:
            g1["inferior"].append(f)
        if float(np.median(a1)) < float(np.median(a2)):
            g1["median_superior"] += 1

    g2 = {"superior": [], "inferior": []}
    for f in base_families:
        a1 = [runs[(f, "base", "A1", s)]["final_best"] for s in range(1, seeds + 1)]
        a3 = [runs[(f, "base", "A3", s)]["final_best"] for s in range(1, seeds + 1) if (f, "base", "A3", s) in runs]
        if len(a3) < seeds:
            continue
        sup, inf = paired_wilcoxon_superiority(a1, a3)
        if sup:
            g2["superior"].append(f)
        if inf:
            g2["inferior"].append(f)

    s2_key = ("hartmann6", "S2")
    g3 = None
    if (s2_key[0], "S2", "A1", 1) in runs and (s2_key[0], "S2", "A2", 1) in runs:
        h1 = [runs[(s2_key[0], "S2", "A1", s)]["band_hits"] for s in range(1, seeds + 1)]
        h2 = [runs[(s2_key[0], "S2", "A2", s)]["band_hits"] for s in range(1, seeds + 1)]
        m2med = float(np.median(h2))
        g3 = {
            "a1_median": float(np.median(h1)),
            "a2_median": m2med,
            "ratio": float(np.median(h1)) / m2med if m2med > 0 else float("inf"),
        }

    s3_key = ("ackley8", "S3")
    g4 = None
    if (s3_key[0], "S3", "A1", 1) in runs:
        delays = [runs[(s3_key[0], "S3", "A1", s)]["drift_trigger_delay"] for s in range(1, seeds + 1)]
        wasted = [runs[(s3_key[0], "S3", "A1", s)]["wasted_after_drift"] or 0 for s in range(1, seeds + 1)]
        triggered = [d is not None for d in delays]
        share = sum(triggered) / len(triggered) if triggered else 0.0
        g4 = {
            "trigger_share": share,
            "delays": delays,
            "wasted_median": float(np.median(wasted)),
            "within_2_rounds_share": sum(1 for d in delays if d is not None and d <= 2) / len(delays),
        }

    dups = sum(r["batch_duplicates"] for r in runs.values() if r["arm"] == "A1")
    s1_key = ("hartmann6", "S1")
    g5 = {"total_batch_duplicates": dups}
    if (s1_key[0], "S1", "A1", 1) in runs:
        rates = [runs[(s1_key[0], "S1", "A1", s)]["feasible_rate"] for s in range(1, seeds + 1)]
        g5["feasible_rate_median"] = float(np.median(rates))

    verdict = {
        "G1": {
            "detail": g1,
            "pass": len(g1["superior"]) >= PROVISIONAL_GATES["G1_min_superior_families"]
            and len(g1["inferior"]) <= PROVISIONAL_GATES["G1_max_inferior_families"]
            and g1["median_superior"] >= PROVISIONAL_GATES["G1_min_median_superior"],
        },
        "G2": {
            "detail": g2,
            "pass": len(g2["superior"]) >= PROVISIONAL_GATES["G2_min_superior_interaction"]
            and len(g2["inferior"]) <= PROVISIONAL_GATES["G2_max_inferior"],
        },
        "G3": {"detail": g3, "pass": g3 is not None and g3["ratio"] >= PROVISIONAL_GATES["G3_band_ratio"]},
        "G4": {
            "detail": g4,
            "pass": g4 is not None
            and g4["within_2_rounds_share"] >= 1 - PROVISIONAL_GATES["G4_max_seed_share_missing"]
            and g4["wasted_median"] <= PROVISIONAL_GATES["G4_max_wasted_multiple_of_B"] * 6,
        },
        "G5": {
            "detail": g5,
            "pass": dups <= PROVISIONAL_GATES["G5_max_batch_duplicates"]
            and g5.get("feasible_rate_median", 1.0) >= PROVISIONAL_GATES["G5_min_feasible_rate"],
        },
    }
    return verdict


def write_report(out: Path, verdict: dict, seeds: int) -> None:
    lines = [
        "# bo-bench 标定报告",
        "",
        f"- 种子数：{seeds}（主门 25 / 开发环 8——#193 协议表）",
        "- 门初值：#193 临别值（本报告 = 首跑标定依据——冻结值回写 `PROVISIONAL_GATES`）",
        "",
        "| 门 | 判定 | 明细 |",
        "| --- | --- | --- |",
    ]
    for name, v in verdict.items():
        lines.append(
            f"| {name} | {'✅ PASS' if v['pass'] else '❌ FAIL'} | `{json.dumps(v['detail'], ensure_ascii=False, default=str)}` |"
        )
    lines += [
        "",
        "## 度量口径（#193 m1–m7）",
        "",
        "- m1/m2 轨迹与 regret：`runs/*.json` 的 cumulative_best_trajectory / final_best",
        "- m3 带命中：band_hits（S2）",
        "- m4 可行产出率：feasible_rate（S1）",
        "- m5 触发时延：drift_trigger_delay（S3）",
        "- m6 漂移浪费：wasted_after_drift（S3）",
        "- m7 批内重复：batch_duplicates（constant_liar 自证 = 0）",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "gates.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=25, help="种子数（主门 25 / 开发环 8）")
    ap.add_argument("--families", nargs="*", help="只跑指定族（开发环）")
    ap.add_argument("--out", type=Path, default=Path(".bench/bo"), help="产出目录")
    args = ap.parse_args()

    runs = run_matrix(args.seeds, args.families, args.out)
    plot_trajectories(runs, args.out, args.seeds)
    verdict = evaluate_gates(runs, args.seeds)
    write_report(args.out, verdict, args.seeds)
    all_pass = all(v["pass"] for v in verdict.values())
    print(json.dumps({k: v["pass"] for k, v in verdict.items()}, indent=1))
    print(f"报告：{args.out / 'report.md'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    from typing import Any

    raise SystemExit(main())
