"""规则引擎 SDK 契约测试：实例门面（register/fire/poll_proposals + pkl 落盘 + 停止合成器）。

seam（公共接口，2026-10 重构定案 + #89 归位——引擎服务于 POMDP，触发上下文 = 观测 o）：

- 规则 = 实现 ``Rule`` 协议的对象（``key`` 身份 + ``propose(o)`` 出题——Rule 定义
  问题、propose 定义对 o 校验后输出的解题选项）——两域词表知识经域适配器聚合
- ``RuleEngine.register(*rules)``：注册（key 撞名 fail-loud 带已注册候选清单）
- ``fire(*keys, o)``：按 key 激活（空 = 全部；未知 key fail-loud），产出
  ``Activation``（形状/类型牙齿住 pydantic 模型构造期；引擎只查实例类型与
  rule_key 一致性），连续 fire 在实例内累积
- ``poll_proposals()``：聚合 + 记分牌序（salience 降序，同分 rule_key 字典序）+ 排空
- ``compose_should_stop(when)``：when(o) 谓词 → action="finish" 提议
  （salience=0 垫底；产物可 pickle 随 rules.pkl 持载）
- 落盘 = 纯函数对（引擎零路径状态）：``load_rule_engine(path)``（缺文件 →
  None；损坏/异类 → raise 带路径）+ ``save_rule_engine(path, engine)``（dumps 后
  ``write_bytes_atomic``）；整读整写的会话门面住 POMDPSession.propose_ctx
  （#89——rule_engine_rwa 已退役，事务语义在 test_pomdp_session 钉住）
"""

import pickle

import pytest
from mflowy.driver.pomdp import Activation
from mflowy.driver.rule import RuleEngine, compose_should_stop
from pydantic import ValidationError


class _EchoRule:
    """测试替身：propose 产出固定 Activation，记录收到的 o。"""

    def __init__(self, key: str, salience: int = 0, action: str | None = None):
        self.key = key
        self._salience = salience
        self._action = action or key

    def propose(self, o):
        return [
            Activation(
                rule_key=self.key,
                action=self._action,
                params={},
                salience=self._salience,
                rationale=f"{self.key} 出题",
            )
        ]


class _CustomRule(_EchoRule):
    """模块级自定义规则替身（pkl 落盘要求：类须可 import 定位）。"""


def _budget_exhausted(o):
    """模块级停止谓词（pickle 要求：模块级函数，非 lambda）。"""
    return o.get("steps", 0) >= o.get("max_steps", 0)


def test_register_fail_loud_on_duplicate_key():
    """撞名 fail-loud：同 key 二次注册报错并给出已注册候选清单（红线条目）。"""
    engine = RuleEngine()
    engine.register(_EchoRule("drop_cols.missing"))
    with pytest.raises(ValueError, match=r"drop_cols\.missing.*已注册"):
        engine.register(_EchoRule("drop_cols.missing"))


def test_register_rejects_bad_key_identity():
    """声明期 fail-loud：非 str / 空 key 当场拒绝，不留到点火期。"""
    engine = RuleEngine()
    with pytest.raises(TypeError, match="key"):
        engine.register(object.__new__(_EchoRule))  # key 属性缺失
    r = _EchoRule("ok")
    r.key = ""
    with pytest.raises(ValueError, match="key"):
        engine.register(r)


# ── fire / poll_proposals ──────────────────────────────────────────────────


def test_fire_all_rules_when_keys_empty_and_accumulates_across_fires():
    """空 keys = 激活全部注册规则；连续 fire 累积，o 原样递给 propose。"""
    seen = []
    r1, r2 = _EchoRule("a", salience=10), _EchoRule("b", salience=20)

    class _Spy:
        key = "spy"

        def propose(self, o):
            seen.append(o)
            return []

    engine = RuleEngine().register(r1, r2, _Spy())
    out1 = engine.fire(o={"state_digest": {"defects": 1}})
    out2 = engine.fire(o={"state_digest": {"defects": 2}})

    assert [a.rule_key for a in out1] == ["a", "b"]  # 本次激活即刻可见
    assert [a.rule_key for a in out2] == ["a", "b"]
    assert seen == [{"state_digest": {"defects": 1}}, {"state_digest": {"defects": 2}}]  # o 原样递给规则
    acts = engine.poll_proposals()
    assert [a.rule_key for a in acts] == ["b", "b", "a", "a"]  # 聚合两次 fire，全局 salience 降序
    assert engine.poll_proposals() == []  # poll 排空


def test_fire_selected_keys_and_fail_loud_on_unknown():
    """显式 keys 只激活命中规则；未知 key 报错带已注册候选清单（fail-loud）。"""
    engine = RuleEngine().register(_EchoRule("a"), _EchoRule("b"))
    out = engine.fire("a", o={})
    assert [x.rule_key for x in out] == ["a"]
    with pytest.raises(ValueError, match=r"nope.*可用.*\['a', 'b'\]"):
        engine.fire("a", "nope", o={})


def test_poll_orders_by_salience_then_rule_key_deterministic():
    """记分牌序：salience 降序，同分按 rule_key 字典序——确定性菜单。"""
    engine = RuleEngine().register(
        _EchoRule("c.same", salience=5),
        _EchoRule("a.same", salience=5),
        _EchoRule("z.top", salience=9),
    )
    acts = engine.fire(o={})
    assert [a.rule_key for a in acts] == ["c.same", "a.same", "z.top"]  # fire 返回注册序
    assert [a.rule_key for a in engine.poll_proposals()] == ["z.top", "a.same", "c.same"]  # poll 记分牌序排空


def test_activation_shape_teeth_live_at_model_construction():
    """形状/类型牙齿住 pydantic 模型构造期（#89）：坏记分牌/空动作/旧 proposal 键当场拒收。"""
    with pytest.raises(ValidationError, match="salience"):
        Activation(rule_key="bad", action="x", params={}, salience=1.5, rationale="r")
    with pytest.raises(ValidationError, match="action"):
        Activation(rule_key="bad", action="", params={}, salience=1, rationale="r")
    with pytest.raises(ValidationError, match="proposal"):
        Activation(rule_key="bad", proposal={"action": "x", "params": {}}, salience=1, rationale="r")


def test_fire_fail_loud_on_foreign_instance_and_rule_key_mismatch():
    """点火期牙齿（模型承接形状后仅剩两条）：非 Activation 实例 / rule_key 与宿主规则不一致。"""
    engine = RuleEngine()
    engine.register(_BadProducer(["裸 dict"]))
    with pytest.raises(TypeError, match="Activation 实例"):
        engine.fire(o={})

    engine2 = RuleEngine()
    engine2.register(_BadProducer([Activation(rule_key="other", action="x", params={}, salience=1, rationale="r")]))
    with pytest.raises(ValueError, match="不一致"):
        engine2.fire(o={})


class _BadProducer:
    def __init__(self, out):
        self.key = "bad"
        self._out = out

    def propose(self, o):
        return self._out


# ── 停止规则合成（#89）──────────────────────────────────────────────────────


def test_compose_should_stop_gate_short_circuit_and_salience_floor():
    """when(o) 假 → 零产出（资格门短路）；真 → 唯一 finish 提议（salience=0 记分牌垫底）。"""
    rule = compose_should_stop(_budget_exhausted)
    assert rule.key == "should_stop"
    assert rule.propose({"steps": 1, "max_steps": 5}) == []
    acts = rule.propose({"steps": 5, "max_steps": 5})
    assert len(acts) == 1
    a = acts[0]
    assert a.action == "finish" and a.params == {} and a.salience == 0
    assert a.rationale == "停止条件成立"

    # 与高 salience 解题规则共排：finish 垫底（真实选项在前）
    engine = RuleEngine().register(_EchoRule("solve", salience=5), compose_should_stop(_budget_exhausted))
    engine.fire(o={"steps": 5, "max_steps": 5})
    assert [(a.action, a.salience) for a in engine.poll_proposals()] == [("solve", 5), ("finish", 0)]


def test_compose_should_stop_custom_key_and_rationale():
    rule = compose_should_stop(_budget_exhausted, key="plateau_stop", rationale="质量平台期")
    assert rule.key == "plateau_stop"
    assert rule.propose({"steps": 9, "max_steps": 9})[0].rationale == "质量平台期"


def test_compose_should_stop_picklable_via_rules_pkl(tmp_path):
    """合成产物随 rules.pkl 持载：模块级谓词可 pickle，跨 save/load 仍出题。"""
    from mflowy.driver.rule import load_rule_engine, save_rule_engine

    engine = RuleEngine().register(compose_should_stop(_budget_exhausted))
    assert pickle.dumps(engine)  # 构造期即可 pickle（lambda 会在 save 期被拒）

    pkl = tmp_path / "rules.pkl"
    save_rule_engine(str(pkl), engine)
    loaded = load_rule_engine(str(pkl))
    assert [a.action for a in loaded.fire(o={"steps": 3, "max_steps": 3})] == ["finish"]


# ── 落盘：load/save 纯函数（引擎零路径状态；整读整写门面住 POMDPSession） ──





def test_load_rule_engine_missing_none_corrupt_fail_loud(tmp_path):
    """load 纯函数契约：缺文件 → None（回退归调用方）；损坏/异类 → raise 带路径
    （不静默重建默认——自定义规则不允许被静默换掉）。"""
    from mflowy.driver.rule import load_rule_engine

    pkl = tmp_path / "rules.pkl"
    assert load_rule_engine(str(pkl)) is None  # 缺文件：None，非异常

    pkl.write_bytes(b"\x80\x04 not-a-pickle")
    with pytest.raises(RuntimeError, match=r"rules\.pkl"):
        load_rule_engine(str(pkl))

    import pickle as _pickle

    pkl.write_bytes(_pickle.dumps({"not": "an engine"}))
    with pytest.raises(TypeError, match="RuleEngine"):
        load_rule_engine(str(pkl))


def test_save_fail_loud_on_unpicklable_rule_zero_file_effect(tmp_path):
    """不可 pickle 规则：save 报错且零文件效果（不落 pkl、不留 tmp 垃圾）。"""
    from mflowy.driver.rule import save_rule_engine

    class _Local:
        key = "x.unpicklable"

        def propose(self, o):  # pragma: no cover - 替身不被调用
            return []

    pkl = tmp_path / "rules.pkl"
    engine = RuleEngine().register(_Local())
    with pytest.raises(AttributeError, match="local"):
        save_rule_engine(str(pkl), engine)
    assert not pkl.exists() and not (tmp_path / f".{pkl.name}.tmp").exists()
