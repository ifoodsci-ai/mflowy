"""测试 utils/jinja.py 模块"""

import pytest
from jinja2 import UndefinedError
from mflowy.utils.jinja import JinjaEnvRegistry, get_yaml_template_env


class TestJinjaEnvRegistry:
    """测试 JinjaEnvRegistry 类"""

    def test_singleton_pattern(self):
        """测试单例模式"""
        registry1 = JinjaEnvRegistry()
        registry2 = JinjaEnvRegistry()
        assert registry1 is registry2

    def test_initialization(self):
        """测试初始化"""
        registry = JinjaEnvRegistry()
        assert registry._initialized is True
        assert isinstance(registry._yaml_envs, dict)

    def test_get_yaml_env_renders_sibling_template_strictly(self, tmp_path):
        """render 消费契约（#224 P3：``env is not None``/hasAttr 同义反复退役）：
        同目录模板可取可渲染 + StrictUndefined（未定义变量报错而非静默空串）"""
        registry = JinjaEnvRegistry()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")
        (tmp_path / "t.j2").write_text("hello {{ who }}")

        env = registry.get_yaml_env(config_file)

        assert env.get_template("t.j2").render(who="mflowy") == "hello mflowy"
        with pytest.raises(UndefinedError):
            env.get_template("t.j2").render()  # StrictUndefined 契约

    def test_yaml_env_cached(self, tmp_path):
        """测试 YAML 环境缓存"""
        registry = JinjaEnvRegistry()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")

        env1 = registry.get_yaml_env(config_file)
        env2 = registry.get_yaml_env(config_file)

        assert env1 is env2

    def test_yaml_env_no_custom_globals(self, tmp_path):
        """测试 YAML 环境不包含 load_yaml（已移至 CLI --env-file）"""
        registry = JinjaEnvRegistry()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")

        env = registry.get_yaml_env(config_file)

        assert "load_yaml" not in env.globals
        assert "combine" not in env.filters


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_get_yaml_template_env(self, tmp_path):
        """便捷函数 = 注册表单源（缓存同 env），render 消费可用（#224 P3）"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")
        (tmp_path / "t.j2").write_text("hi {{ x }}")

        env = get_yaml_template_env(config_file)

        assert env is JinjaEnvRegistry().get_yaml_env(config_file)  # 同一缓存实例
        assert env.get_template("t.j2").render(x=1) == "hi 1"
