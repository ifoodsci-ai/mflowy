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


class _EchoRule:
    """测试替身：propose 产出固定 Activation，记录收到的 R。"""

    def __init__(self, key: str, salience: int = 0, action: str | None = None):
        self.key = key
        self._salience = salience
        self._action = action or key

    def propose(self, R):
        return [
            {
                "rule_key": self.key,
                "proposal": {"action": self._action, "params": {}},
                "salience": self._salience,
                "rationale": f"{self.key} 出题",
            }
        ]


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
