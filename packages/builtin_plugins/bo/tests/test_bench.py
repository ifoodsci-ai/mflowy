"""bo-bench 契约测试（#231）：函数族已知最优 / f_pre 编码 / harness 全流程冒烟 /
采样器换装词表（A2/A3 臂缝单源）。

冒烟预算：单臂 3 轮（≈10s）；全矩阵标定归 scripts/_bo_bench.py 人肉门，不进 make test。
"""

from pathlib import Path

import numpy as np
import pytest
from mflowy.bo.bench import FAMILIES, SurrogateFPre, run_bench_arm
from mflowy.bo.bench.functions import Hartmann6, Rosenbrock4
from mflowy.bo.bench.harness import _lhs_sample, make_scenario


class TestFunctions:
    @pytest.mark.parametrize("name", sorted(FAMILIES))
    def test_known_optimum_is_lower_bound(self, name):
        """已知最优 = 下界锚：随机样本不显著优于 known_optimum（声明不被野破）——
        regret 归一分母的合法性；可达性由各族专测钉（解析点/构造点评估）。"""
        fam = FAMILIES[name]
        samples = [fam.true_f(r) for r in _lhs_sample(fam.dims, 200, seed=7)]
        assert min(samples) >= fam.known_optimum - 1e-6, (name, min(samples), fam.known_optimum)

    def test_rosenbrock_global_point(self):
        f = Rosenbrock4()
        assert f.true_f({f"x{i}": 1.0 for i in range(1, 5)}) == pytest.approx(0.0, abs=1e-9)

    def test_hartmann_dimensionality(self):
        f = Hartmann6()
        assert len(f.dims) == 6
        assert f.true_f({f"x{i}": 0.5 for i in range(1, 7)}) > f.known_optimum  # 中心劣于最优（非退化）

    def test_quadratic_optimum_at_mu(self):
        from mflowy.bo.bench.functions import QuadraticCross6

        f = QuadraticCross6()
        at_mu = {c: float(v) for c, v in zip(sorted(f.dims), f.mu, strict=True)}
        assert f.true_f(at_mu) == pytest.approx(0.0, abs=1e-9)  # 解析峰可达

    def test_denaturation_peak_reachable(self):
        f = FAMILIES["denaturation_cliff"]
        peak = {"temperature": 55.0, "ph": 7.5, "concentration": 0.40, "enzyme": "E2"}
        assert f.true_f(peak) == pytest.approx(-1.0, abs=1e-6)  # E2 峰构造已知

    def test_mixed_families_have_categorical(self):
        for name in ("morphed_hartmann", "denaturation_cliff"):
            assert any(s["kind"] == "discrete" for s in FAMILIES[name].dims.values())


class TestFPre:
    def test_interval_is_constant_width(self):
        rows = _lhs_sample(FAMILIES["quadratic_cross6"].dims, 120, seed=3)
        y = np.array([FAMILIES["quadratic_cross6"].true_f(r) for r in rows])
        fp = SurrogateFPre().fit(__import__("pandas").DataFrame(rows), y)
        p1, w1 = fp.predict(rows[0])
        p2, w2 = fp.predict(rows[1])
        assert w1 == w2 and w1 > 0  # 恒宽区间契约

    def test_mixed_encoding(self):
        rows = _lhs_sample(FAMILIES["denaturation_cliff"].dims, 120, seed=3)
        y = np.array([FAMILIES["denaturation_cliff"].true_f(r) for r in rows])
        fp = SurrogateFPre().fit(__import__("pandas").DataFrame(rows), y)
        point, width = fp.predict(rows[0])  # 含字符串类别——不抛即编码面成立
        assert np.isfinite(point) and width > 0


class TestAskPathSampler:
    def test_ask_path_honors_engine(self, tmp_path, monkeypatch):
        """#231 修复钉测：ask 路径 sampler = engine 声明（#175 mv+cl 显式开真生效——
        基准 G1 失败抓出的生产缺陷：attr 挂了、sampler 没接，落 optuna 缺省 TPE）。"""
        import pandas as pd
        from mflowy.bo import study as bo_study
        from mflowy.bo import tools as bo_tools
        from mflowy.bo.bench.functions import FAMILIES
        from mflowy.bo.bench.harness import _lhs_sample

        fam = FAMILIES["hartmann6"]
        rows = _lhs_sample(fam.dims, 20, seed=1)
        df = pd.DataFrame(rows)
        df["y"] = [fam.true_f(r) for r in rows]
        hist = tmp_path / "w.parquet"
        df.to_parquet(hist)
        init = bo_tools.bo_load_or_start(
            study_name="sampler-pin",
            model_run_id="x",
            data=str(hist),
            target_column="y",
            direction="minimize",
            search_space=fam.dims,
        )
        st, _, _ = bo_tools._open_with_dims(init["study_db"])
        assert st.sampler._multivariate is True and st.sampler._constant_liar is True  # type: ignore[attr-defined]
        # bench 臂换装（engine.sampler=random）同面生效
        st2 = bo_study.open_study(Path(init["study_db"]))
        engine = dict(st2.user_attrs[bo_study.ATTR_ENGINE])
        engine["sampler"] = "random"
        st2.set_user_attr(bo_study.ATTR_ENGINE, engine)
        st3, _, _ = bo_tools._open_with_dims(init["study_db"])
        from optuna.samplers import RandomSampler

        assert isinstance(st3.sampler, RandomSampler)


class TestSamplerKinds:
    def test_vocab_and_fail_loud(self):
        from mflowy.bo import study

        assert study.SAMPLER_KINDS == ("tpe", "random", "tpe_univariate")
        with pytest.raises(ValueError, match="sampler 须为"):
            study.build_sampler({"sampler": "gibbs", "multivariate": True, "constant_liar": True, "seed": 1}, 0)

    def test_arm_samplers_build(self):
        from mflowy.bo import study
        from optuna.samplers import RandomSampler, TPESampler

        assert isinstance(
            study.build_sampler({"sampler": "random", "multivariate": True, "constant_liar": True, "seed": 1}, 0),
            RandomSampler,
        )
        uni = study.build_sampler(
            {"sampler": "tpe_univariate", "multivariate": True, "constant_liar": True, "seed": 1}, 0
        )
        assert isinstance(uni, TPESampler)


class TestHarnessSmoke:
    def test_one_arm_full_loop(self, tmp_path):
        """全流程真实工具面冒烟：init → ask → prune → diversity → tell（+信号批末）；
        m7 批内重复 = 0（constant_liar 自证）。"""
        rec = run_bench_arm(
            scenario="base", family_name="quadratic_cross6", seed=11, arm="A1", workdir=tmp_path, rounds=3
        )
        assert len(rec["wet_values"]) == 3 * 6  # R 轮 × B 批全湿
        assert rec["batch_duplicates"] == 0
        assert rec["cumulative_best_trajectory"][-1] == rec["final_best"]
        assert 0 < rec["feasible_rate"] <= 1

    def test_random_arm_patch(self, tmp_path):
        rec = run_bench_arm(
            scenario="base", family_name="quadratic_cross6", seed=12, arm="A2", workdir=tmp_path, rounds=2
        )
        assert len(rec["wet_values"]) == 2 * 6

    def test_scenario_gates(self, tmp_path):
        sc = make_scenario("S2", "hartmann6")
        assert sc.direction == "match" and sc.band is not None and sc.band[1] > sc.band[0]
        sc1 = make_scenario("S1", "hartmann6")
        assert sc1.hard_rule is not None
        with pytest.raises(ValueError, match="未知场景"):
            make_scenario("S9", "hartmann6")
