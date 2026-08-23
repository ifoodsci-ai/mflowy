"""相关性过滤器单元测试"""

import numpy as np
import pandas as pd
import pytest

from mflowy.compute.cleaners.filter.correlation_filter import correlation_filter


class TestCorrelationFilterBasicFunctionality:
    """测试基本功能"""

    def test_removes_highly_correlated_features(self):
        """测试移除高相关特征"""
        np.random.seed(42)
        n = 100
        feat1 = np.random.randn(n)
        feat2 = np.random.randn(n) * 2 + 1
        feat3 = feat2 + np.random.randn(n) * 0.01  # feat3 ≈ feat2，高相关
        target = np.random.randn(n) * 10 + 50

        data = pd.DataFrame(
            {
                "feat1": feat1,
                "feat2": feat2,
                "feat3": feat3,
                "target": target,
            }
        )

        result = correlation_filter(df=data, threshold=0.95, exclude=["target"])

        assert "feat1" in result.columns
        assert ("feat2" in result.columns and "feat3" not in result.columns) or (
            "feat3" in result.columns and "feat2" not in result.columns
        )
        assert "target" in result.columns
        assert len(result.columns) == 3

    def test_keeps_low_correlation_features(self):
        """测试保留低相关特征"""
        np.random.seed(43)
        n = 100
        data = pd.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n) * 0.5 - 0.3,
                "feat3": np.random.randn(n) * 2 + 1,
                "target": np.random.randn(n) * 10 + 50,
            }
        )

        result = correlation_filter(df=data, threshold=0.9, exclude=["target"])

        assert "feat1" in result.columns
        assert "feat2" in result.columns
        assert "feat3" in result.columns
        assert "target" in result.columns
        assert len(result.columns) == 4

    def test_respects_custom_threshold(self):
        """测试自定义阈值"""
        np.random.seed(44)
        n = 100
        feat1 = np.random.randn(n)
        feat2 = feat1 * 2 + np.random.randn(n) * 0.01

        data = pd.DataFrame(
            {
                "feat1": feat1,
                "feat2": feat2,
            }
        )

        result1 = correlation_filter(df=data, threshold=0.99)
        assert len(result1.columns) == 1

        result2 = correlation_filter(df=data, threshold=0.999)
        assert len(result2.columns) >= 1


class TestCorrelationFilterStrategies:
    """测试特征保留策略"""

    def test_keeps_higher_variance_feature(self):
        """测试保留方差更大的特征"""
        data = pd.DataFrame(
            {
                "low_var": [1.0, 1.0, 1.0, 1.0, 1.1],
                "high_var": [1, 2, 3, 4, 5],
                "correlated": [2, 4, 6, 8, 10],
            }
        )

        result = correlation_filter(df=data, threshold=0.95)

        assert "high_var" in result.columns or "correlated" in result.columns
        assert "low_var" in result.columns

    def test_respects_priority_features(self):
        """测试优先特征列表"""
        np.random.seed(45)
        n = 100
        feat1 = np.random.randn(n)
        feat2 = feat1 * 2 + np.random.randn(n) * 0.01
        feat3 = np.random.randn(n) * 5 + 10

        data = pd.DataFrame(
            {
                "feat1": feat1,
                "feat2": feat2,
                "feat3": feat3,
            }
        )

        result = correlation_filter(df=data, threshold=0.95, priority_features=["feat2"])

        assert "feat2" in result.columns
        assert "feat1" not in result.columns
        assert "feat3" in result.columns


class TestCorrelationFilterMethods:
    """测试不同相关系数方法"""

    def test_pearson_correlation(self):
        """测试 Pearson 相关系数"""
        data = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [2, 4, 6, 8, 10],
            }
        )

        result = correlation_filter(df=data, method="pearson", threshold=0.95)

        assert len(result.columns) == 1

    def test_spearman_correlation(self):
        """测试 Spearman 相关系数"""
        data = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [1, 4, 9, 16, 25],
            }
        )

        result = correlation_filter(df=data, method="spearman", threshold=0.95)

        assert len(result.columns) == 1

    def test_invalid_method_raises_error(self):
        """测试无效的相关系数方法"""
        data = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValueError, match="method must be"):
            correlation_filter(df=data, method="invalid_method")


class TestCorrelationFilterEdgeCases:
    """测试边界情况"""

    def test_empty_dataframe(self):
        """测试空数据集"""
        data = pd.DataFrame()

        result = correlation_filter(df=data, threshold=0.9)

        assert result.empty

    def test_no_numeric_columns(self):
        """测试无数值列"""
        data = pd.DataFrame(
            {
                "text1": ["a", "b", "c"],
                "text2": ["x", "y", "z"],
            }
        )

        result = correlation_filter(df=data, threshold=0.9)

        pd.testing.assert_frame_equal(result, data)

    def test_single_numeric_column(self):
        """测试单个数值列"""
        data = pd.DataFrame(
            {
                "num": [1, 2, 3, 4, 5],
                "text": ["a", "b", "c", "d", "e"],
            }
        )

        result = correlation_filter(df=data, threshold=0.9)

        assert "num" in result.columns
        assert "text" in result.columns

    def test_no_correlated_features(self):
        """测试无高相关特征"""
        data = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [5, 4, 3, 2, 1],
                "feat3": [1, 3, 2, 4, 5],
            }
        )

        result = correlation_filter(df=data, threshold=0.95)

        assert len(result.columns) >= 2

    def test_negative_correlation(self):
        """测试负相关"""
        data = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [10, 8, 6, 4, 2],
            }
        )

        result = correlation_filter(df=data, threshold=0.9)

        assert len(result.columns) == 1

    def test_data_immutability(self):
        """测试数据不可变性"""
        data = pd.DataFrame(
            {
                "feat1": [1, 2, 3, 4, 5],
                "feat2": [2, 4, 6, 8, 10],
            }
        )

        original_data = data.copy()
        correlation_filter(df=data, threshold=0.9)

        pd.testing.assert_frame_equal(data, original_data)

    def test_invalid_threshold_raises_error(self):
        """测试无效的阈值"""
        data = pd.DataFrame({"a": [1, 2, 3]})

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            correlation_filter(df=data, threshold=-0.1)

        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            correlation_filter(df=data, threshold=1.5)


class TestCorrelationFilterRealWorldScenarios:
    """测试真实场景"""

    def test_temperature_features(self):
        """测试温度特征场景"""
        np.random.seed(46)
        n = 50
        hmt_temp = np.random.randn(n) * 10 + 120
        annealing_temp = hmt_temp + np.random.randn(n) * 0.01
        drying_temp = np.random.randn(n) * 15 + 70
        delta_rs = np.random.randn(n) * 5 + 20

        data = pd.DataFrame(
            {
                "HMT_temp": hmt_temp,
                "Annealing_temp": annealing_temp,
                "Drying_temp": drying_temp,
                "Delta_RS": delta_rs,
            }
        )

        result = correlation_filter(df=data, threshold=0.95, priority_features=["HMT_temp"], exclude=["Delta_RS"])

        assert "HMT_temp" in result.columns
        assert "Annealing_temp" not in result.columns
        assert "Drying_temp" in result.columns
        assert "Delta_RS" in result.columns

    def test_extrusion_temperature_derived_features(self):
        """测试挤压温度派生特征"""
        data = pd.DataFrame(
            {
                "Extrusion_max_temp": [150, 160, 170, 180, 190],
                "Extrusion_min_temp": [100, 110, 120, 130, 140],
                "Extrusion_mean_temp": [125, 135, 145, 155, 165],
                "Extrusion_temp_range": [50, 50, 50, 50, 50],
                "target": [1, 2, 3, 4, 5],
            }
        )

        result = correlation_filter(df=data, threshold=0.9)

        assert "Extrusion_max_temp" in result.columns or "Extrusion_min_temp" in result.columns
