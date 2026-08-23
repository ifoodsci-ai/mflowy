"""单元测试：训练重构后的新增函数（get_param_space, is_supported, shared utilities）"""

import pytest

try:
    import optuna  # noqa: F401  仅作可用性检查（get_sampler 内部 lazy import）
except (ImportError, OSError) as e:
    pytest.skip(f"Optuna not available: {e}", allow_module_level=True)


# ---------------------------------------------------------------------------
# get_sampler error path
# ---------------------------------------------------------------------------
class TestGetSamplerErrors:
    def test_unknown_sampler_raises(self):
        from mflowy.utils.study import get_sampler

        with pytest.raises(KeyError, match="未知的优化方法"):
            get_sampler("nonexistent")
