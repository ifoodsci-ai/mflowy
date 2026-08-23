"""测试 utils/jinja.py 模块"""

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

    def test_get_yaml_env(self, tmp_path):
        """测试获取 YAML 环境"""
        registry = JinjaEnvRegistry()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")

        env = registry.get_yaml_env(config_file)

        assert env is not None
        assert hasattr(env, "globals")

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
        """测试获取 YAML 模板环境"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: data")

        env = get_yaml_template_env(config_file)
        assert env is not None
