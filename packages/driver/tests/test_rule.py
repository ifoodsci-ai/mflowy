"""规则引擎 SDK 契约测试：实例门面（register/fire/poll_proposals + pkl 落盘）。

seam（公共接口，2026-10 重构定案）：

- 规则 = 实现 ``Rule`` 协议的对象（``key`` 身份 + ``propose(R)`` 出题）——
  两域词表知识（clean ``.repair`` / model 族与 FE）经域适配器聚合到引擎实例上
- ``RuleEngine.register(*rules)``：注册（key 撞名 fail-loud 带候选清单）
- ``fire(*keys, R)``：按 key 激活（空 = 全部；未知 key fail-loud），产出
  ``Activation``，连续 fire 在实例内累积
- ``poll_proposals()``：聚合 + 记分牌序（salience 降序，同分 rule_key 字典序）+ 排空
- ``open_engine(path, default=...)``：盘上有 pkl 则加载，否则从默认实例复制
  （激活态清空）；with 正常退出原子落盘，异常不落盘
"""

import pytest
from mflowy.driver.rule import Activation


class _EchoRule:
    """测试替身：propose 产出固定 Activation，记录收到的 R。"""

    def __init__(self, key: str, salience: int = 0, action: str | None = None):
        self.key = key
        self._salience = salience
        self._action = action or key

    def propose(self, R):
        return [
            Activation(
                rule_key=self.key,
                proposal={"action": self._action, "params": {}},
                salience=self._salience,
                rationale=f"{self.key} 出题",
            )
        ]


class _CustomRule(_EchoRule):
    """模块级自定义规则替身（pkl 落盘要求：类须可 import 定位）。"""


def test_register_fail_loud_on_duplicate_key():
    """撞名 fail-loud：同 key 二次注册报错并给出已注册候选清单（红线条目）。"""
    from mflowy.driver.rule import RuleEngine

    engine = RuleEngine()
    engine.register(_EchoRule("drop_cols.missing"))
    with pytest.raises(ValueError, match=r"drop_cols\.missing.*已注册"):
        engine.register(_EchoRule("drop_cols.missing"))


def test_register_rejects_bad_key_identity():
    """声明期 fail-loud：非 str / 空 key 当场拒绝，不留到点火期。"""
    from mflowy.driver.rule import RuleEngine

    engine = RuleEngine()
    with pytest.raises(TypeError, match="key"):
        engine.register(object.__new__(_EchoRule))  # key 属性缺失
    r = _EchoRule("ok")
    r.key = ""
    with pytest.raises(ValueError, match="key"):
        engine.register(r)


# ── fire / poll_proposals ──────────────────────────────────────────────────


def test_fire_all_rules_when_keys_empty_and_accumulates_across_fires():
    """空 keys = 激活全部注册规则；连续 fire 累积，R 原样递给 propose。"""
    from mflowy.driver.rule import RuleEngine

    seen = []
    r1, r2 = _EchoRule("a", salience=10), _EchoRule("b", salience=20)

    class _Spy:
        key = "spy"

        def propose(self, R):
            seen.append(R)
            return []

    engine = RuleEngine().register(r1, r2, _Spy())
    out1 = engine.fire(R={"defects": 1})
    out2 = engine.fire(R={"defects": 2})

    assert [a.rule_key for a in out1] == ["a", "b"]  # 本次激活即刻可见
    assert [a.rule_key for a in out2] == ["a", "b"]
    assert seen == [{"defects": 1}, {"defects": 2}]  # R 原样递给规则
    acts = engine.poll_proposals()
    assert [a.rule_key for a in acts] == ["b", "b", "a", "a"]  # 聚合两次 fire，全局 salience 降序
    assert engine.poll_proposals() == []  # poll 排空


def test_fire_selected_keys_and_fail_loud_on_unknown():
    """显式 keys 只激活命中规则；未知 key 报错带已注册候选清单（fail-loud）。"""
    from mflowy.driver.rule import RuleEngine

    engine = RuleEngine().register(_EchoRule("a"), _EchoRule("b"))
    out = engine.fire("a", R={})
    assert [x.rule_key for x in out] == ["a"]
    with pytest.raises(ValueError, match=r"nope.*可用.*\['a', 'b'\]"):
        engine.fire("a", "nope", R={})


def test_poll_orders_by_salience_then_rule_key_deterministic():
    """记分牌序：salience 降序，同分按 rule_key 字典序——确定性菜单。"""
    from mflowy.driver.rule import RuleEngine

    engine = RuleEngine().register(
        _EchoRule("c.same", salience=5),
        _EchoRule("a.same", salience=5),
        _EchoRule("z.top", salience=9),
    )
    acts = engine.fire(R={})
    assert [a.rule_key for a in acts] == ["c.same", "a.same", "z.top"]  # fire 返回注册序
    assert [a.rule_key for a in engine.poll_proposals()] == ["z.top", "a.same", "c.same"]  # poll 记分牌序排空


def test_fire_fail_loud_on_malformed_activation():
    """点火期 fail-loud：坏 Activation 形状/类型/键集 当场拒绝，不静默吞。"""
    from mflowy.driver.rule import RuleEngine

    class _Bad:
        def __init__(self, out):
            self.key = "bad"
            self._out = out

        def propose(self, R):
            return self._out

    engine = RuleEngine()
    engine.register(_Bad(["裸 dict"]))
    with pytest.raises(TypeError, match="Activation 实例"):
        engine.fire(R={})

    engine2 = RuleEngine()
    engine2.register(_Bad([Activation("other", {"action": "x", "params": {}}, 1, "r")]))
    with pytest.raises(ValueError, match="不一致"):
        engine2.fire(R={})

    engine3 = RuleEngine()
    engine3.register(_Bad([Activation("bad", {"action": "x"}, 1, "r")]))
    with pytest.raises(ValueError, match="proposal"):
        engine3.fire(R={})

    engine4 = RuleEngine()
    engine4.register(_Bad([Activation("bad", {"action": "x", "params": {}}, 1.5, "r")]))
    with pytest.raises(TypeError, match="salience"):
        engine4.fire(R={})


# ── 落盘：save/load/copy + open_engine ─────────────────────────────────────


def test_open_engine_fresh_path_copies_default_and_saves_on_exit(tmp_path):
    """盘上无 pkl：从默认实例复制（激活态清空）；with 正常退出原子落盘。"""
    from mflowy.driver.rule import RuleEngine, open_engine

    default = RuleEngine().register(_EchoRule("a"), _EchoRule("b"))
    default.fire(R={})  # 默认实例被误点火也不得带进会话实例

    pkl = tmp_path / "rules.pkl"
    with open_engine(pkl, default=default) as engine:
        assert sorted(engine.rules) == ["a", "b"]
        assert engine.poll_proposals() == []  # 激活态清空
        engine.fire(R={})
    assert pkl.exists() and not pkl.with_name(pkl.name + ".tmp").exists()  # 原子：无残留 tmp


def test_open_engine_loads_custom_rules_and_pending_activations(tmp_path):
    """盘上有 pkl：加载实例——运行期注册的自定义规则与未 poll 激活态跨调用存活。"""
    from mflowy.driver.rule import RuleEngine, open_engine

    default = RuleEngine().register(_EchoRule("a"))
    pkl = tmp_path / "rules.pkl"

    with open_engine(pkl, default=default) as engine:
        engine.register(_CustomRule("custom.x", salience=7))
        engine.fire("custom.x", R={})  # 不 poll——挂起态落盘

    with open_engine(pkl, default=default) as engine:
        assert "custom.x" in engine.rules
        acts = engine.poll_proposals()
        assert [a.rule_key for a in acts] == ["custom.x"]  # 挂起激活态恢复

    with open_engine(pkl, default=default) as engine:
        assert engine.poll_proposals() == []  # 上一轮已 poll 排空并落盘


def test_open_engine_exception_no_save(tmp_path):
    """with 内异常：异常透传且不落盘（事务性——失败调用零副作用）。"""
    import pytest as _pytest

    from mflowy.driver.rule import RuleEngine, open_engine

    default = RuleEngine().register(_EchoRule("a"))
    pkl = tmp_path / "rules.pkl"
    with _pytest.raises(RuntimeError, match="boom"):
        with open_engine(pkl, default=default) as engine:
            engine.register(_EchoRule("custom"))
            raise RuntimeError("boom")
    assert not pkl.exists()


def test_open_engine_fail_loud_on_corrupt_pkl(tmp_path):
    """损坏 pkl：fail-loud 带路径与成因，不静默重建默认。"""
    from mflowy.driver.rule import open_engine

    pkl = tmp_path / "rules.pkl"
    pkl.write_bytes(b"\x80\x04 not-a-pickle")
    with pytest.raises(RuntimeError, match="rules.pkl"):
        open_engine(pkl, default=None).__enter__()
