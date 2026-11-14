"""BOCard —— bo 域先验卡契约（pydantic 单源，#178/#184 定名）。

字段两层分组（同一性判据 = 影响 ``study.ask`` 结果的字段，#178 M1）：

- **空间组**（变更 = 重建 study）：``search_space``（每维三 kind + 边界/choices）+
  ``direction`` + ``seed``（#175 Q4：sampler 随 seed 建）。搜索维边界 =
  DomainCard ``expected_ranges`` 直取（物理域唯一权威在上游；缺列时卡自持该维边界
  ——卡自持边界这一事实本身即缺口的表达，#178 R4）。
- **策略组**（变更不重建、不升版本号——卡 = 可变工件，无实例版本纪律）：hard 规则
  表达式树 / 按类常量表 / band / 容差 / 信号门限 / 批缺省 / 典型测量变异。

**工具零卡耦合**（#180 R3）：bo 工具不读本卡——门限/规则/容差全由 agent 读卡后以
参数或判断进入；本模块是卡的书写契约（bo-prior 剧本产 ``<task_dir>/BOCard.yaml``）
与机器校验面（未知谓词/未知列/未知表键/类型不合一律 fail-loud，红线 1）。

结构性防错（#178 R5 构造性隔离）：软约束唯一形态 = ``band`` 标量，无表达式面——
「变性阈值永不得进软约束」在构造上不可能。
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from mflowy.utils.python_script_security_scan import scan_security
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── 空间组：搜索维三 kind（#178 R2；编码声明归空间组承载但不进同一性 M1）──────


class ContinuousDim(BaseModel):
    """连续维：low/high（= expected_ranges 直取）+ dtype + scale。log 维须 low > 0。"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["continuous"]
    low: float
    high: float
    dtype: Literal["int", "float"] = "float"
    scale: Literal["linear", "log"] = "linear"
    # 描述符/编码声明：f_pre 消费面（agent/剧本知识）——不影响 ask 不进同一性（M1），工具不消费
    encoding: str | None = None


class DiscreteDim(BaseModel):
    """离散维：类别清单（优化变量入特征空间——酶列即典型，#176）。"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["discrete"]
    choices: list[float | int | str | bool] = Field(min_length=1)
    encoding: str | None = None


class CombinatorialDim(BaseModel):
    """组合维：v1 薄 payload——组合 token 清单（序列变体等结构承载转上游 #178 R6）。"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["combinatorial"]
    choices: list[str] = Field(min_length=1)
    encoding: str | None = None
    note: str | None = None


type SpaceDim = Annotated[Union[ContinuousDim, DiscreteDim, CombinatorialDim], Field(discriminator="kind")]

# ── 策略组：hard 规则表达式树（JSON 树——Callable 过不了 MCP 通道，D14）────────
#
# 谓词词表：比较 le/lt/ge/gt/eq、between、查表 lookup_le/lookup_lt/lookup_ge/
# lookup_gt（表键 + 常量名 + 安全裕度）；跨特征算术 add/sub/mul/div/min/max；
# 逻辑组合 all/any/not。未知谓词 fail-loud 带候选清单（pydantic 判别联合天然拒收）。


class Arith(BaseModel):
    """跨特征算术：args ≥ 2，元素为特征名（str）/ 常量（number）/ 常量引用 / 嵌套算术。"""

    model_config = ConfigDict(extra="forbid")

    op: Literal["add", "sub", "mul", "div", "min", "max"]
    args: list[Operand] = Field(min_length=2)


class ConstRef(BaseModel):
    """非数值常量引用（类别字面量等）——bare str 一律是特征名，字面量须走 ``{"const": …}``。"""

    model_config = ConfigDict(extra="forbid")

    const: str


type Operand = Union[Arith, ConstRef, float, int, str]


class ComparisonRule(BaseModel):
    """比较谓词：left OP right（操作数 = 特征名 / 常量 / 算术表达式）。"""

    model_config = ConfigDict(extra="forbid")

    op: Literal["le", "lt", "ge", "gt", "eq"]
    left: Operand
    right: Operand


class BetweenRule(BaseModel):
    """区间谓词：low ≤ feature ≤ high。"""

    model_config = ConfigDict(extra="forbid")

    op: Literal["between"]
    feature: str
    low: float
    high: float


class LookupRule(BaseModel):
    """查表谓词（#176 R5）：feature OP lookup_tables[table][key_feature 取值][constant] ± margin。

    安全裕度 margin 收缩可行域：le/lt 侧减去、ge/gt 侧加上（例：变性温度上限
    ``temperature lookup_le ENZYME_DENAT[enzyme].T_max - margin``）。
    """

    model_config = ConfigDict(extra="forbid")

    op: Literal["lookup_le", "lookup_lt", "lookup_ge", "lookup_gt"]
    feature: str
    table: str
    key_feature: str
    constant: str
    margin: float = 0.0


type AtomRule = Union[ComparisonRule, BetweenRule, LookupRule]
type _Rule = Union["AllRule", "AnyRule", "NotRule", AtomRule]


class AllRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["all"]
    rules: list[_Rule] = Field(min_length=1)


class AnyRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["any"]
    rules: list[_Rule] = Field(min_length=1)


class NotRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["not"]
    rule: _Rule


type Rule = Union[AllRule, AnyRule, NotRule, AtomRule]

# 判别联合按 op 键判别（各 op 值域不相交）
type DiscriminatedRule = Annotated[_Rule, Field(discriminator="op")]
type DiscriminatedDim = SpaceDim


class SignalWindow(BaseModel):
    """小窗口门：最近 K 条推荐里 m 条负向即触发（m 须 < 负向累计门，否则冗余）。"""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(ge=1)
    m: int = Field(ge=1)


class BOCard(BaseModel):
    """BO 先验卡（正式名 BOCard，#184；_Avoid_: 「先验卡」单指 bo 卡——那是 DomainCard）。

    pydantic 契约单源（``extra="forbid"``）；卡实例 = YAML 工件（MCP 通道友好）。
    同一性字段 = ``search_space`` + ``direction``（+ 工具侧 seed，#175 Q4）——
    变更即重建 study（bo_load_or_init 迁移式开局承接）；策略组字段可后改。
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    # 引用上游卡（C3 书写约定：剧本承载逐条确认，无机制）
    inherited_from: str | None = None

    # ── 空间组（同一性：变更 = 重建 study）────────────────────────────────────
    target_column: str = Field(min_length=1)
    direction: Literal["maximize", "minimize", "match"]
    search_space: dict[str, DiscriminatedDim] = Field(min_length=1)

    # ── 策略组（变更不重建、不升版本号）──────────────────────────────────────
    # 业务目标带：match 必填；max/min 可选作业务门。软约束唯一形态 = 此标量（构造性隔离）
    band: tuple[float, float] | None = None
    hard_rules: list[DiscriminatedRule] = Field(default_factory=list)
    # 按类常量表：表键 → 类别值 → 常量名 → 值（#176 R5）
    lookup_tables: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    # python 逃生舱：``validate(row: pd.DataFrame) -> bool`` 源码串（单行候选 DataFrame——
    # search_input cross_rules 同 seam：utils scan_security 白名单 + 签名约束）
    python_rule: str | None = None
    # 两容差（目标单位，可空）：|y−ŷ| ≤ close → 接近；≥ severe → 严重不符；之间不计
    tolerance_close: float | None = Field(default=None, ge=0)
    tolerance_severe: float | None = Field(default=None, ge=0)
    # 信号门限（推荐级计数制，#177 R4；缺省 = 草案值；窗口门缺省关）
    signal_positive_gate: int = Field(default=100, ge=1)
    signal_negative_gate: int = Field(default=3, ge=1)
    signal_window: SignalWindow | None = None
    # 批大小缺省（实验室容量）：必填、可逐轮覆写（无代码级缺省——bo-prior 问出）
    batch_size_default: int = Field(ge=1)
    # 典型测量变异 w（ŷ±w 统计观测参考，#178 M3：机器消费方 = agent 统计观测）
    typical_measurement_variation: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_card(self) -> BOCard:
        # match 必填 band（方向三态的目标带语义）
        if self.direction == "match" and self.band is None:
            raise ValueError("direction=match 必填 band（lo, hi）——目标带是 match 语义的一部分")
        if self.band is not None and self.band[0] >= self.band[1]:
            raise ValueError(f"band 需 lo < hi，收到 {self.band}")

        # 维度结构：连续维 low < high；log 维 low > 0
        for name, dim in self.search_space.items():
            if isinstance(dim, ContinuousDim):
                if dim.low >= dim.high:
                    raise ValueError(f"搜索维 '{name}' 需 low < high，收到 [{dim.low}, {dim.high}]")
                if dim.dtype == "int" and (dim.low != int(dim.low) or dim.high != int(dim.high)):
                    raise ValueError(f"搜索维 '{name}' dtype=int 需整值边界，收到 [{dim.low}, {dim.high}]")
                if dim.scale == "log" and dim.low <= 0:
                    raise ValueError(f"搜索维 '{name}' scale=log 需 low > 0，收到 {dim.low}")
            elif isinstance(dim, DiscreteDim) and len(set(map(str, dim.choices))) != len(dim.choices):
                raise ValueError(f"离散维 '{name}' choices 含重复值")

        # 规则树：未知列/未知表键/未知常量名 fail-loud 带候选清单（红线 1）
        self._walk_rules_for_validation(self.hard_rules)

        # 容差序：close < severe（之间不计的区间非空）
        if (
            self.tolerance_close is not None
            and self.tolerance_severe is not None
            and self.tolerance_close >= self.tolerance_severe
        ):
            raise ValueError(
                f"tolerance_close 须小于 tolerance_severe（之间不计），收到 {self.tolerance_close} ≥ {self.tolerance_severe}"
            )

        # 窗口门：m < 负向累计门否则冗余（#178 R3）
        if self.signal_window is not None and self.signal_window.m >= self.signal_negative_gate:
            raise ValueError(
                f"signal_window.m 须小于 signal_negative_gate（否则累计门先响、窗口门冗余），收到 m={self.signal_window.m} ≥ {self.signal_negative_gate}"
            )

        # python 逃生舱：书写即扫（validate(row: pd.DataFrame) -> bool 签名约束 + 危险引用白名单）
        if self.python_rule is not None:
            import pandas as pd

            scan_security(self.python_rule, func_name="validate", args={"row": pd.DataFrame}, returns=bool)

        return self

    def _walk_rules_for_validation(self, rules: list[Rule]) -> None:
        """规则树结构校验（fail-loud）：特征引用 ⊆ 搜索维；查表引用闭合。"""
        dims = set(self.search_space.keys())

        def check_operand(op: Operand) -> None:
            if isinstance(op, Arith):
                for a in op.args:
                    check_operand(a)
            elif isinstance(op, str):
                if op not in dims:
                    raise ValueError(f"规则引用未知特征 '{op}'——候选：{sorted(dims)}")
            # ConstRef / 数值常量：无引用面，免检

        def check_rule(r: Rule) -> None:
            if isinstance(r, ComparisonRule):
                check_operand(r.left)
                check_operand(r.right)
            elif isinstance(r, BetweenRule):
                if r.feature not in dims:
                    raise ValueError(f"规则引用未知特征 '{r.feature}'——候选：{sorted(dims)}")
            elif isinstance(r, LookupRule):
                if r.feature not in dims:
                    raise ValueError(f"规则引用未知特征 '{r.feature}'——候选：{sorted(dims)}")
                if r.key_feature not in dims:
                    raise ValueError(f"查表谓词引用未知键特征 '{r.key_feature}'——候选：{sorted(dims)}")
                table = self.lookup_tables.get(r.table)
                if table is None:
                    raise ValueError(f"查表谓词引用未知表键 '{r.table}'——候选：{sorted(self.lookup_tables) or '无'}")
                for row_key, constants in table.items():
                    if r.constant not in constants:
                        raise ValueError(
                            f"查表谓词引用表 '{r.table}' 行 '{row_key}' 的未知常量 '{r.constant}'"
                            f"——候选：{sorted(constants) or '无'}"
                        )
                # 键特征取值须能命中表行（离散/组合维的 choices ⊆ 表行键）
                key_dim = self.search_space[r.key_feature]
                key_values = key_dim.choices if isinstance(key_dim, DiscreteDim | CombinatorialDim) else None
                if key_values is not None:
                    missing = [v for v in key_values if str(v) not in table]
                    if missing:
                        raise ValueError(
                            f"查表谓词表 '{r.table}' 缺键特征 '{r.key_feature}' 取值的行：{[str(v) for v in missing]}"
                            f"——表行：{sorted(table)}"
                        )
            elif isinstance(r, AllRule | AnyRule):
                for sub in r.rules:
                    check_rule(sub)
            elif isinstance(r, NotRule):
                check_rule(r.rule)

        for rule in rules:
            check_rule(rule)

    # ── YAML 工件面（bo-prior 剧本收口落 <task_dir>/BOCard.yaml）──────────────

    @classmethod
    def from_yaml(cls, text: str) -> BOCard:
        import yaml

        return cls.model_validate(yaml.safe_load(text))

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
