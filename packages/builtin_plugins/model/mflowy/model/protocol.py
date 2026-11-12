"""建模域 CVCard（原 ProtocolCard，#63 全链改名）：评估协议卡的冻结载体。

与 DomainCard（数据先验，mflowy.data.types）对称——两卡都是 episode 内冻结的
POMDP 输入：先验卡管"数据是什么"，协议卡管"对局怎么打"（task/metric/CV/预算/
结构化候选）。「先验」为两卡通用家族（#169/#170）：先验卡 = 数据域先验；
协议卡的 external_baseline = 建模域先验（前沿研究的领域先验知识——结构化对标）。
换协议 = board 作废 + test 重触，只能弃局开新 episode。

candidates = ``[{family, baseline: bool, space_hints: {param: 值 | [lo, hi]}, seed}]``
（research-review 剧本产物）：baseline 恰一（propose 时第一）；space_hints ⊆ 族
签名声明空间（标量 = pin、区间 = 收窄——文献知识注入搜索起点）；每候选默认随机
种子（搜索期内冻结——换 seed = 换目标 = study 重开，判据在回路）。
"""

from __future__ import annotations

from dataclasses import dataclass

from mflowy.data.types import TASKTYPE
from mflowy.model.model.types import MetricName
from mflowy.utils.constants import RANDOM_STATE

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

# 外部基线权威度白名单（#169/#170——同构 #112 DomainCard.source 三值白名单）
_EXTERNAL_BASELINE_SOURCES = ("literature", "internal_history", "manual")


def _validate_external_baseline(eb, task: TASKTYPE) -> dict | None:
    """外部基线对标结构校验（#169/#170——建模域先验）：数值 value + 口径 metric
    （须与本任务指标同域）+ 出处 source_text + 权威度 source 白名单，fail-loud。
    None = 未对标（空 dict 走缺键 fail-loud——宁可拒绝，不静默归一）；自由文本
    str 已退役（BREAKING，见 CHANGELOG）。"""
    if eb is None:
        return None
    if not isinstance(eb, dict):
        raise ValueError(
            f"external_baseline 须为结构化 dict {{value, metric, source_text, source}}"
            f"（#170 起自由文本退役，BREAKING），got {type(eb).__name__}: {eb!r}"
        )
    required = {"value", "metric", "source_text", "source"}
    missing = sorted(required - set(eb))
    if missing:
        raise ValueError(
            f"external_baseline 缺键 {missing}（结构化对标 = 数值 value + 口径 metric + 出处 source_text + 权威度 source）"
        )
    unknown = sorted(set(eb) - required)
    if unknown:
        raise ValueError(f"external_baseline 未知键 {unknown}（可用：{sorted(required)}）")
    import math

    value = eb["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"external_baseline.value 须为有限数值，got {value!r}")
    allowed = _TASK_METRICS[task]
    if str(eb["metric"]) not in allowed:
        raise ValueError(
            f"external_baseline.metric `{eb['metric']}` 与任务 {task.name.lower()} 不相容"
            f"（可用: {sorted(allowed)}）——对照口径须与本任务指标同域"
        )
    if not str(eb["source_text"]).strip():
        raise ValueError("external_baseline.source_text 须非空（出处——文献引用 / 内部实验记录）")
    if eb["source"] not in _EXTERNAL_BASELINE_SOURCES:
        raise ValueError(f"external_baseline.source 须为 {'/'.join(_EXTERNAL_BASELINE_SOURCES)}，得到 {eb['source']!r}")
    return {
        "value": float(value),
        "metric": str(eb["metric"]),
        "source_text": str(eb["source_text"]),
        "source": eb["source"],
    }


def _validate_hints(family: str, hints) -> dict:
    """hints ⊆ 签名声明空间（fail-loud 带可用面清单）：标量 pin / 区间收窄 → JSON 形态。"""
    from mflowy.model.model.study import ContinuousSpace, DiscreteSpace
    from mflowy.model.model.utils import normalize_space, space_to_json
    from mflowy.model.registry import family_face

    face = family_face(family)  # 词表派生面单源（#141 起冻结面，零活反射）
    out: dict = {}
    for param, value in (hints or {}).items():
        if param not in face:
            raise ValueError(f"候选 {family} 的 space_hints[{param}] 不在签名声明空间（可用：{sorted(face)}）")
        declared = face[param]
        if isinstance(declared, (ContinuousSpace, DiscreteSpace)):
            out[param] = space_to_json(normalize_space(param, declared, value))
        elif isinstance(value, (list, tuple, dict)):
            raise ValueError(
                f"候选 {family} 的 space_hints[{param}] 缺省是静态常量 {declared!r}——静态面走标量 pin，不可开空间"
            )
        elif isinstance(value, (int, float, str, bool)):
            if param == "random_state":
                raise ValueError(f"候选 {family} 的 seed 走候选的 seed 字段（搜索环境参数），不是 space_hints")
            out[param] = value
        else:
            raise ValueError(
                f"候选 {family} 的 space_hints[{param}] 声明非法（标量 = pin / 区间 = 收窄），got {value!r}"
            )
    return out


@dataclass
class CVCard:
    """建模会话的评估协议。字段与 modeling_session_start 的冻结 protocol 分量一一对应。

    candidates 空 = 全部已声明族开放（缺省，字典序、首个为 baseline）；research-review
    剧本产出收窄清单（文献先验第一轮淘汰 + baseline 锚点 + 空间 hints + 候选 seed）。
    预算 = 单一 max_steps（π 自行分配：跑哪些候选、搜多深——n_trials 是深度旋钮）。
    """

    task: TASKTYPE
    cv: int  # inner CV 折数——缺省仅由 build 签名持有（#83 单源，字段无第二份拷贝）
    max_steps: int  # 单一预算：只有计算动作（run_hpo/final_train_test）吃步——缺省仅由 build 签名持有
    metric: MetricName | None = None  # None = 任务自动选（infer_metric）
    candidates: tuple[dict, ...] = ()  # 空 = 全部已声明族开放（首个 baseline）
    goal: str = ""  # 建模成功判据（业务口径，区别于 survey 的研究目标）
    # 外部基线对标（建模域先验面，#169/#170）：结构化 dict {value, metric, source_text,
    # source}——校验单源 _validate_external_baseline；渲染面等「波动主导差距」挂层
    external_baseline: dict | None = None

    def as_dict(self) -> dict:
        return {
            "task": self.task.name.lower(),  # agent 友好名（regression/classification）
            "metric": str(self.metric) if self.metric else None,
            "cv": self.cv,
            "max_steps": self.max_steps,
            "candidates": [dict(c) for c in self.candidates],
            "goal": self.goal,
            "external_baseline": dict(self.external_baseline) if self.external_baseline is not None else None,
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
        candidates=None,
        goal: str = "",
        external_baseline: dict | None = None,
    ) -> CVCard:
        """问卷答案 → 协议卡（四类相容校验 fail-loud 的单一事实源）。

        ① task↔target dtype（分类目标须低基数/类别、回归目标须数值）；
        ② metric↔task（指标任务域匹配，None 自动选）；
        ③ cv↔数据事实（2–10 折；分类时 cv ≤ 最小类计数）；
        ④ 预算与候选机械校验（max_steps ≥ 1；候选 ⊆ 已声明族目录、baseline 恰一、
        hints ⊆ 签名声明空间）——引擎不判断先验的科学性。

        external_baseline 结构化对标（#169/#170 建模域先验）由
        `_validate_external_baseline` 校验（口径同域 + 权威度白名单 fail-loud）。
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
                f"分类目标 `{target}` 为连续数值列（nunique={y.nunique()})——分类目标应为低基数/类别列，或改答 regression"
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
            raise ValueError(f"预算 max_steps 须 ≥ 1（只有计算动作吃步，π 自行分配），got {max_steps}")

        from mflowy.model.registry import families

        catalog = families()
        raw_cands = (
            list(candidates)
            if candidates
            else [{"family": f, "baseline": i == 0} for i, f in enumerate(sorted(catalog))]
        )
        if not raw_cands:
            raise ValueError(f"candidates 不可为空清单（可用族目录：{sorted(catalog)}）")
        normalized: list[dict] = []
        for c in raw_cands:
            fam = c.get("family")
            if fam not in catalog:
                raise ValueError(f"候选 {fam!r} 不在已声明族目录 {sorted(catalog)}（.mdp_family 声明者）")
            entry = {
                "family": fam,
                "baseline": bool(c.get("baseline", False)),
                "space_hints": _validate_hints(fam, c.get("space_hints")),
                "seed": c.get("seed", RANDOM_STATE),
            }
            if not isinstance(entry["seed"], int) or isinstance(entry["seed"], bool):
                raise ValueError(f"候选 {fam} 的 seed 须为 int（搜索环境参数），got {entry['seed']!r}")
            normalized.append(entry)
        n_baseline = sum(1 for c in normalized if c["baseline"])
        if n_baseline != 1:
            raise ValueError(
                f"baseline 恰一（research-review 标记锚点候选——propose 时第一），实得 {n_baseline} 个"
                f"（候选：{[c['family'] for c in normalized]}）"
            )

        return cls(
            task=t,
            metric=m,
            cv=cv,
            max_steps=max_steps,
            candidates=tuple(normalized),
            goal=goal,
            external_baseline=_validate_external_baseline(external_baseline, t),
        )
