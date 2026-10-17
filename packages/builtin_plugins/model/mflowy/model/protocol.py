"""建模域协议卡：对局规则的冻结载体（model-brief 问卷产物，内容命名）。

与 DomainCard（数据先验，mflowy.data.types）对称——两卡都是 episode 内冻结的
MDP 输入：先验卡管"数据是什么"，协议卡管"对局怎么打"（task/metric/CV/预算/
候选族/资源先验）。换协议 = board 作废 + test 重触，只能弃局开新 episode。
"""

from __future__ import annotations

from dataclasses import dataclass

from mflowy.data.types import TASKTYPE
from mflowy.model.model.types import MetricName

_TASK_METRICS: dict[str, set[str]] = {
    TASKTYPE.REGRESSION: {"mae", "rmse", "r2", "mape"},
    TASKTYPE.CLASSIFICATION: {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "kappa",
        "auc_roc",
        "pr_auc",
        "logloss",
        "mlogloss",
        "brier",
        "ece",
    },
}


@dataclass
class ProtocolCard:
    """建模会话的对局规则。字段与 modeling_session_start 的冻结 protocol 分量一一对应。

    candidate_families 空 = 全部已声明族开放（缺省）；research-review 剧本可产出
    收窄清单（文献先验第一轮淘汰）。预算 = 单一 max_steps（π 自行分配：跑哪些
    候选、搜多深——n_trials 是深度旋钮；资源 = 签名参数，#33 资源机制全退）。
    """

    task: TASKTYPE
    metric: MetricName | None = None  # None = 任务自动选（infer_metric）
    cv: int = 5
    max_steps: int = 10  # 单一预算：每动作（run/FE）吃一步——FE 实验成本 2 步，缺省上调
    candidate_families: tuple[str, ...] = ()  # 空 = 全部已声明族开放
    goal: str = ""  # 建模成功判据（业务口径，区别于 survey 的研究目标）
    external_baseline: str = ""  # 外部基线对标（自由文本，board 对照面）

    def as_dict(self) -> dict:
        return {
            "task": self.task.name.lower(),  # agent 友好名（regression/classification）
            "metric": str(self.metric) if self.metric else None,
            "cv": self.cv,
            "max_steps": self.max_steps,
            "candidate_families": list(self.candidate_families),
            "goal": self.goal,
            "external_baseline": self.external_baseline,
        }

    @classmethod
    def build(
        cls,
        df,
        *,
        target: str,
        task: str,
        metric: str | None = None,
        cv: int = 5,
        max_steps: int = 10,
        candidate_families=None,
        goal: str = "",
        external_baseline: str = "",
    ) -> ProtocolCard:
        """问卷答案 → 协议卡（四类相容校验 fail-loud 的单一事实源）。

        ① task↔target dtype（分类目标须低基数/类别、回归目标须数值）；
        ② metric↔task（指标任务域匹配，None 自动选）；
        ③ cv↔数据事实（2–10 折；分类时 cv ≤ 最小类计数）；
        ④ 预算与候选机械校验（max_steps ≥ 1；候选 ⊆ 已声明族目录）——
        引擎不判断先验的科学性。
        """
        import pandas as pd

        if target not in df.columns:
            raise ValueError(f"目标列 `{target}` 不在数据列中: {list(df.columns)}")
        from mflowy.driver.config import parse_enum

        y = df[target].dropna()
        raw = str(task).strip()
        if raw.upper() in TASKTYPE.__members__:  # 大小写名（REGRESSION/regression）归一
            raw = raw.upper()
        t = parse_enum(TASKTYPE, raw)  # 双形式：值(regressor)/名(REGRESSION)皆收
        is_num = pd.api.types.is_numeric_dtype(y)
        if t == TASKTYPE.CLASSIFICATION and is_num and y.nunique() > 20:
            raise ValueError(
                f"分类目标 `{target}` 为连续数值列（nunique={y.nunique()}）——分类目标应为低基数/类别列，或改答 regression"
            )
        if t == TASKTYPE.REGRESSION and not is_num:
            raise ValueError(
                f"回归目标 `{target}` 为非数值列（dtype={y.dtype}）——回归目标须数值列，或改答 classification"
            )

        if metric is not None:
            allowed = _TASK_METRICS[t]
            if str(metric) not in allowed:
                raise ValueError(f"指标 `{metric}` 与任务 {t} 不相容（{t} 可用: {sorted(allowed)}）")
            m = MetricName(str(metric))
        else:
            from mflowy.model.model.utils import infer_metric

            m = infer_metric(t)

        if not 2 <= cv <= 10:
            raise ValueError(f"cv 折数须在 [2, 10]，got {cv}")
        if t == TASKTYPE.CLASSIFICATION:
            min_cls = int(y.value_counts().min())
            if cv > min_cls:
                raise ValueError(f"cv={cv} 超过最小类计数 {min_cls}（分层折切不出）——降低折数或补数据")

        if max_steps < 1:
            raise ValueError(f"预算 max_steps 须 ≥ 1（每 run 吃一步，π 自行分配），got {max_steps}")

        from mflowy.model.registry import families

        catalog = families()
        candidates = tuple(candidate_families or ())
        unknown = [f for f in candidates if f not in catalog]
        if unknown:
            raise ValueError(f"candidate_families {unknown} 不在已声明族目录 {sorted(catalog)}（.mdp_family 声明者）")

        return cls(
            task=t,
            metric=m,
            cv=cv,
            max_steps=max_steps,
            candidate_families=candidates,
            goal=goal,
            external_baseline=external_baseline,
        )
