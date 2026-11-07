"""model.study 单元测试：纯骨架 search + 三场景封装（ml / dl / inverse）。

场景知识固化处：ml = tpe 无剪枝（K 折无诚实中间值）；dl = tpe+Hyperband
（资源轴增量评估的长训练）；inverse = 单目标 tpe / 多目标 NSGAII（pruner 与
多目标不兼容，恒无 pruner）。sampler 由场景封装构造，get_sampler 已退役。
"""

import optuna
import pytest
from mflowy.model.model.study import ContinuousSpace, search, search_dl, search_inverse, search_ml


@pytest.fixture(autouse=True)
def _quiet_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)


class TestSkeleton:
    def test_search_passes_sampler_and_pruner_through(self):
        """纯骨架：sampler/pruner/directions 全透传，本函数零策略。"""
        sampler = optuna.samplers.RandomSampler(seed=7)
        pruner = optuna.pruners.NopPruner()

        def obj(trial, x):  # noqa: ARG001
            return x

        study = search({"x": ContinuousSpace(0.0, 1.0)}, obj, 3, sampler=sampler, pruner=pruner, direction="maximize")
        assert len(study.trials) == 3
        assert study.sampler is sampler and study.pruner is pruner


class TestSearchML:
    def test_tpe_no_pruner(self):
        """ml 场景：tpe + 无 pruner——K 折无逐折独立中间值，剪枝不诚实。"""

        def obj(trial, x):
            return x * x

        study = search_ml({"x": ContinuousSpace(0.0, 2.0)}, obj, 5, direction="maximize")
        assert isinstance(study.sampler, optuna.samplers.TPESampler)
        assert isinstance(study.pruner, optuna.pruners.NopPruner)


class TestSearchDL:
    def test_tpe_hyperband_and_pruning_works(self):
        """dl 场景：tpe + Hyperband——劣质 trial 沿资源轴上报后被剪。"""
        pruned = []

        def obj(trial, x):
            bad = x < 0.5
            for step in (2, 4, 8):
                trial.report(0.0 if bad else x, step=step)
                if trial.should_prune():
                    pruned.append(trial.number)
                    raise optuna.TrialPruned()
            return x

        study = search_dl({"x": ContinuousSpace(0.0, 1.0)}, obj, 12, max_resource=8, direction="maximize", seed=1)
        assert isinstance(study.sampler, optuna.samplers.TPESampler)
        assert isinstance(study.pruner, optuna.pruners.HyperbandPruner)
        assert pruned, "劣质 trial 应被 Hyperband 剪掉"

    def test_min_resource_formula(self):
        """min_resource 单源：max(1, cap//4)——dl 封装与曲线检查点共用。"""
        from mflowy.model.model.study import min_resource_of

        assert min_resource_of(30) == 7 and min_resource_of(3) == 1 and min_resource_of(8) == 2


class TestSearchInverse:
    def test_single_direction_tpe(self):
        def obj(trial, x):
            return x

        study = search_inverse({"x": ContinuousSpace(0.0, 1.0)}, obj, 4, direction="maximize", seed=42)
        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_multi_direction_nsgaii(self):
        """多目标：NSGAII + 恒无 pruner（pruner 与多目标不兼容）。"""

        def obj(trial, x):
            return (x, 1 - x)

        study = search_inverse(
            {"x": ContinuousSpace(0.0, 1.0)},
            obj,
            6,
            directions=["maximize", "maximize"],
            seed=42,
        )
        assert isinstance(study.sampler, optuna.samplers.NSGAIISampler)
        assert isinstance(study.pruner, optuna.pruners.NopPruner)
