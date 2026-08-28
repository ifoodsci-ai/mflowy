"""driver 内核契约：@handler 的 params_phaser 由插件侧自注册，内核零数据结构知识。

- 未注册（None）：参数透传，不做任何转换
- 注册了：StepConf 解析期经 get_post_init 调用（见 discover/StepConf 接线）
转换行为（Enum/搜索空间）的测试在 builtin_plugins 的 test_params_phaser.py。
"""

from mflowy.driver.config import StepConf
from mflowy.driver.handler import handler


def test_no_phaser_params_passthrough():
    """无 phaser 的 handler：StepConf.params 原样保留（内核不猜测结构）"""

    @handler()
    def plain(x=1, **_): ...

    assert plain.convert_params is None
    conf = StepConf(type="load", module="plain", params={"x": {"shape": "raw-dict"}})
    assert conf.params == {"x": {"shape": "raw-dict"}}


def test_phaser_applied_at_conf_parse(fake_plugins):
    """注册了 phaser：StepConf.__post_init__ 解析期调用"""

    def double_factory(fn):
        def double(params):
            return {k: v * 2 for k, v in params.items()}

        return double

    @handler(params_phaser=double_factory)
    def scaled(n=1, **_): ...

    fake_plugins.setdefault("load", {})["scaled"] = scaled
    conf = StepConf(type="load", module="scaled", params={"n": 21})
    assert conf.params == {"n": 42}


def test_phaser_marker_survives_for_discover():
    """convert_params 属性恒存在（None 亦然）——discover 双属性 marker 校验不破"""

    @handler()
    def a(**_): ...

    @handler(params_phaser=lambda fn: lambda p: p)
    def b(**_): ...

    assert hasattr(a, "convert_params") and hasattr(b, "convert_params")
