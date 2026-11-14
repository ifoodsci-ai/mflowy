"""BOCard 契约测试（#178/#184）：空间组/策略组校验 fail-loud + 规则求值器。

结构面（pydantic 判别联合/extra=forbid）与语义面（match 必填 band、同一性词表、
容差序、窗口门冗余校验、python 逃生舱 scan_security）分层钉住；规则求值器与卡的
结构校验共用同一棵树（单源走查）。
"""

import pytest
from mflowy.bo.rules import evaluate_hard_rules
from mflowy.bo.types import BOCard
from pydantic import ValidationError


def _base_card(**overrides) -> dict:
    card = {
        "target_column": "yield",
        "direction": "maximize",
        "search_space": {
            "temperature": {"kind": "continuous", "low": 20, "high": 90},
            "enzyme": {"kind": "discrete", "choices": ["WT", "E1"]},
        },
        "batch_size_default": 6,
    }
    card.update(overrides)
    return card


# ── 空间组 ─────────────────────────────────────────────────────────────────────


def test_match_requires_band():
    with pytest.raises(ValidationError, match="match 必填 band"):
        BOCard.model_validate(_base_card(direction="match"))
    BOCard.model_validate(_base_card(direction="match", band=[40.0, 60.0]))  # 给了即合法


def test_band_ordering_and_dim_bounds():
    with pytest.raises(ValidationError, match="lo < hi"):
        BOCard.model_validate(_base_card(direction="match", band=[60.0, 40.0]))
    with pytest.raises(ValidationError, match="low < high"):
        BOCard.model_validate(_base_card(search_space={"t": {"kind": "continuous", "low": 90, "high": 20}}))
    with pytest.raises(ValidationError, match="log 需 low > 0"):
        BOCard.model_validate(
            _base_card(search_space={"t": {"kind": "continuous", "low": 0, "high": 9, "scale": "log"}})
        )
    with pytest.raises(ValidationError, match="choices 含重复"):
        BOCard.model_validate(_base_card(search_space={"e": {"kind": "discrete", "choices": ["WT", "WT"]}}))


def test_extra_forbid_and_three_kinds():
    with pytest.raises(ValidationError):
        BOCard.model_validate(_base_card(unknown_field=1))
    card = BOCard.model_validate(
        _base_card(
            search_space={
                "t": {"kind": "continuous", "low": 1, "high": 9, "dtype": "int"},
                "p": {"kind": "continuous", "low": 1e-4, "high": 1.0, "scale": "log"},
                "e": {"kind": "discrete", "choices": [1, 2]},
                "v": {"kind": "combinatorial", "choices": ["A1B", "A2B"], "note": "薄 payload"},
            }
        )
    )
    assert set(card.search_space) == {"t", "p", "e", "v"}


# ── 策略组 ─────────────────────────────────────────────────────────────────────


def test_rule_validation_fail_loud():
    with pytest.raises(ValidationError, match="未知特征 'ph'"):
        BOCard.model_validate(_base_card(hard_rules=[{"op": "le", "left": "ph", "right": 7}]))
    with pytest.raises(ValidationError, match="未知表键 'denat'"):
        BOCard.model_validate(
            _base_card(
                hard_rules=[
                    {
                        "op": "lookup_le",
                        "feature": "temperature",
                        "table": "denat",
                        "key_feature": "enzyme",
                        "constant": "t_max",
                    }
                ]
            )
        )
    tables = {"denat": {"WT": {"t_max": 50.0}}}
    with pytest.raises(ValidationError, match="未知常量"):
        BOCard.model_validate(
            _base_card(
                lookup_tables=tables,
                hard_rules=[
                    {
                        "op": "lookup_le",
                        "feature": "temperature",
                        "table": "denat",
                        "key_feature": "enzyme",
                        "constant": "t_min",
                    }
                ],
            )
        )
    with pytest.raises(ValidationError, match="缺键特征 'enzyme' 取值的行"):
        # E1 不在表行 → 键特征取值命中校验红（fail-loud 带表行清单）
        BOCard.model_validate(
            _base_card(
                lookup_tables=tables,
                hard_rules=[
                    {
                        "op": "lookup_le",
                        "feature": "temperature",
                        "table": "denat",
                        "key_feature": "enzyme",
                        "constant": "t_max",
                    }
                ],
            )
        )


def test_tolerance_and_window_gates():
    with pytest.raises(ValidationError, match="tolerance_close 须小于"):
        BOCard.model_validate(_base_card(tolerance_close=5.0, tolerance_severe=3.0))
    with pytest.raises(ValidationError, match="窗口门冗余"):
        BOCard.model_validate(_base_card(signal_window={"k": 10, "m": 3}))  # m == 负向累计门 3
    BOCard.model_validate(_base_card(signal_window={"k": 10, "m": 2}))  # m < 3 合法
    with pytest.raises(ValidationError, match="batch_size_default"):
        BOCard.model_validate({k: v for k, v in _base_card().items() if k != "batch_size_default"})


def test_python_rule_scanned_at_validation():
    bad = "import pandas as pd\n\ndef validate(row: pd.DataFrame) -> bool:\n    return eval('1')\n"  # 危险引用
    with pytest.raises(ValidationError):
        BOCard.model_validate(_base_card(python_rule=bad))
    good = (
        "import pandas as pd\n\ndef validate(row: pd.DataFrame) -> bool:\n    return row['temperature'].iloc[0] > 25\n"
    )
    BOCard.model_validate(_base_card(python_rule=good))


def test_yaml_roundtrip():
    card = BOCard.model_validate(
        _base_card(
            direction="match",
            band=[40.0, 60.0],
            tolerance_close=1.0,
            tolerance_severe=5.0,
            typical_measurement_variation=2.0,
        )
    )
    assert BOCard.from_yaml(card.to_yaml()) == card


# ── 求值器（与结构校验共用单源树）──────────────────────────────────────────────


@pytest.fixture
def card() -> BOCard:
    return BOCard.model_validate(
        _base_card(
            direction="match",
            band=[40.0, 60.0],
            lookup_tables={"denat": {"WT": {"t_max": 50.0}, "E1": {"t_max": 70.0}}},
            hard_rules=[
                {
                    "op": "lookup_le",
                    "feature": "temperature",
                    "table": "denat",
                    "key_feature": "enzyme",
                    "constant": "t_max",
                    "margin": 5,
                },
                {"op": "between", "feature": "temperature", "low": 20, "high": 90},
                {
                    "op": "all",
                    "rules": [
                        {"op": "le", "left": {"op": "add", "args": ["temperature", 10]}, "right": 95},
                        {"op": "not", "rule": {"op": "eq", "left": "enzyme", "right": {"const": "E1"}}},
                    ],
                },
            ],
        )
    )


def test_lookup_margin_shrinks_feasible_region(card):
    # WT t_max=50，margin 5 → temperature ≤ 45 才可行
    assert evaluate_hard_rules(card, {"temperature": 46.0, "enzyme": "WT"}) == ["hard_rules[0]"]
    assert evaluate_hard_rules(card, {"temperature": 45.0, "enzyme": "WT"}) == []


def test_logic_not_and_any(card):
    # not eq enzyme E1 → E1 违 hard_rules[2]
    assert evaluate_hard_rules(card, {"temperature": 30.0, "enzyme": "E1"}) == ["hard_rules[2]"]
    assert evaluate_hard_rules(card, {"temperature": 30.0, "enzyme": "WT"}) == []


def test_arithmetic_and_between(card):
    assert evaluate_hard_rules(card, {"temperature": 95.0, "enzyme": "WT"}) == [
        "hard_rules[0]",  # lookup：95 > WT 上限 45
        "hard_rules[1]",  # between：95 越界
        "hard_rules[2]",  # 算术：95+10 > 95
    ]


def test_evaluator_fail_loud_on_unknown_row_feature(card):
    with pytest.raises(ValueError, match="不在候选行"):
        evaluate_hard_rules(card, {"temperature": 30.0})
    with pytest.raises(ValueError, match="无行 'E2'"):
        evaluate_hard_rules(card, {"temperature": 30.0, "enzyme": "E2"})


def test_python_escape_hatch(card):
    rule = (
        "import pandas as pd\n\ndef validate(row: pd.DataFrame) -> bool:\n    return row['temperature'].iloc[0] < 40\n"
    )
    card = card.model_copy(update={"python_rule": rule})
    assert evaluate_hard_rules(card, {"temperature": 44.0, "enzyme": "WT"}) == [
        "python_rule"
    ]  # hard 规则全过、仅 python 违例
    assert evaluate_hard_rules(card, {"temperature": 30.0, "enzyme": "WT"}) == []
