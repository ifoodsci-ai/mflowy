"""测试 _flatten_params 参数展平"""

from mflowy.driver.builtin_middleware import _flatten_params


class TestFlattenParams:
    def test_scalar_types(self):
        result = _flatten_params({"a": "hello", "b": 1, "c": 2.0, "d": True})
        assert result == {"a": "hello", "b": 1, "c": 2.0, "d": True}

    def test_skip_none(self):
        result = _flatten_params({"a": 1, "b": None, "c": "x"})
        assert result == {"a": 1, "c": "x"}

    def test_dict_flatten(self):
        result = _flatten_params({"space": {"lr": 0.1, "epochs": 10}})
        assert result == {"space.lr": 0.1, "space.epochs": 10}

    def test_nested_dict(self):
        result = _flatten_params({"a": {"b": {"c": 1}}})
        assert result == {"a.b.c": 1}

    def test_tuple_keeps_parens(self):
        result = _flatten_params({"lr": (0.1, 0.2)})
        assert result == {"lr": "(0.1, 0.2)"}

    def test_list_json_dumps(self):
        result = _flatten_params({"layers": [64, 128, 256]})
        assert result == {"layers": "[64, 128, 256]"}

    def test_dict_with_tuple_value(self):
        result = _flatten_params({"params_space": {"learning_rate": (0.1, 0.2)}})
        assert result == {"params_space.learning_rate": "(0.1, 0.2)"}

    def test_empty_dict(self):
        assert _flatten_params({}) == {}

    def test_mixed(self):
        result = _flatten_params(
            {
                "epochs": 100,
                "space": {"lr": (0.01, 0.1), "batch_size": 32},
                "note": None,
                "tags": ["exp1", "exp2"],
            }
        )
        assert result == {
            "epochs": 100,
            "space.lr": "(0.01, 0.1)",
            "space.batch_size": 32,
            "tags": '["exp1", "exp2"]',
        }


class TestMlflowKeyEscape:
    """#151/#152：mlflow 键名字符集（字母数字/_-.空格:斜杠）外字符 → -XX 确定性
    转义——单位 token 键（%/°C/公斤）经 DAG 遥测面不再炸；值面不受限。"""

    def test_percent_token_key_escaped(self):
        assert _flatten_params({"factors": {"%": 1}}) == {"factors.-25": 1}

    def test_non_ascii_tokens_escaped_per_utf8_byte(self):
        out = _flatten_params({"to": "°C", "u": {"公斤": 1000}})
        assert out == {"to": "°C", "u.-e5-85-ac-e6-96-a4": 1000}  # 值原样、键按 UTF-8 字节转义

    def test_safe_keys_untouched_and_composition(self):
        out = _flatten_params({"columns": {"pct": {"to": "%", "factors": {"kg": 1000}}}})
        assert out == {"columns.pct.to": "%", "columns.pct.factors.kg": 1000}  # 合法键零改写、值不动
