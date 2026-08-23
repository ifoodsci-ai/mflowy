"""测试 simple_cv handler"""

import numpy as np
import pandas as pd
import pytest

from mflowy.compute.cross_validation.simple_cv import simple_cv


def _split(X, **kwargs):
    """适配新 handler：返回 Indices 元组列表"""
    return list(simple_cv(X, y=None, **kwargs))


class TestSimpleCV:
    def test_split_returns_indices(self):
        X = pd.DataFrame({"feature": range(50)})
        folds = _split(X, train_ratio=0.8, random_state=42)

        assert len(folds) == 1
        train_idx, val_idx, test_idx = folds[0]
        assert isinstance(train_idx, np.ndarray)
        assert val_idx is None  # 无验证集时为 None
        assert isinstance(test_idx, np.ndarray)

    def test_split_sizes(self):
        X = pd.DataFrame({"feature": range(100)})
        folds = _split(X, train_ratio=0.8, random_state=42)

        _, _, test_idx = folds[0]
        expected_test_size = 100 - int(100 * 0.8)
        assert len(test_idx) == pytest.approx(expected_test_size, abs=2)

    def test_split_with_small_dataset(self):
        X = pd.DataFrame({"feature": range(10)})
        folds = _split(X, train_ratio=0.7, random_state=42)

        assert len(folds) == 1
        train_idx, _, test_idx = folds[0]
        assert len(train_idx) >= 1
        assert len(test_idx) >= 1

    def test_split_reproducibility(self):
        X = pd.DataFrame({"feature": range(50)})
        folds1 = _split(X, train_ratio=0.8, random_state=42)
        folds2 = _split(X, train_ratio=0.8, random_state=42)

        assert np.array_equal(folds1[0][0], folds2[0][0])
        assert np.array_equal(folds1[0][2], folds2[0][2])

    def test_split_no_overlap(self):
        X = pd.DataFrame({"feature": range(100)})
        train_idx, _, test_idx = _split(X, train_ratio=0.8, random_state=42)[0]

        all_indices = np.concatenate([train_idx, test_idx])
        assert len(all_indices) == 100
        assert len(np.unique(all_indices)) == 100

    def test_invalid_train_ratio(self):
        X = pd.DataFrame({"feature": range(10)})
        with pytest.raises(ValueError, match="train_ratio"):
            _split(X, train_ratio=1.5)

    def test_zero_train_ratio(self):
        X = pd.DataFrame({"feature": range(10)})
        with pytest.raises(ValueError, match="train_ratio"):
            _split(X, train_ratio=0.0)


class TestSimpleCVWithVal:
    def test_val_split_returns_three_sets(self):
        X = pd.DataFrame({"feature": range(100)})
        train_idx, val_idx, test_idx = _split(X, train_ratio=0.6, val_ratio=0.2, random_state=42)[0]

        assert val_idx is not None
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0

    def test_val_split_no_overlap_and_complete(self):
        X = pd.DataFrame({"feature": range(100)})
        train_idx, val_idx, test_idx = _split(X, train_ratio=0.6, val_ratio=0.2, random_state=42)[0]

        all_indices = np.concatenate([train_idx, val_idx, test_idx])
        assert len(all_indices) == 100
        assert len(np.unique(all_indices)) == 100

    def test_val_split_approximate_ratios(self):
        X = pd.DataFrame({"feature": range(1000)})
        train_idx, val_idx, test_idx = _split(X, train_ratio=0.6, val_ratio=0.2, random_state=42)[0]

        assert len(train_idx) == pytest.approx(600, abs=10)
        assert len(val_idx) == pytest.approx(200, abs=10)
        assert len(test_idx) == pytest.approx(200, abs=10)

    def test_val_split_reproducibility(self):
        X = pd.DataFrame({"feature": range(100)})
        folds1 = _split(X, train_ratio=0.6, val_ratio=0.2, random_state=42)
        folds2 = _split(X, train_ratio=0.6, val_ratio=0.2, random_state=42)

        assert np.array_equal(folds1[0][0], folds2[0][0])
        assert np.array_equal(folds1[0][1], folds2[0][1])
        assert np.array_equal(folds1[0][2], folds2[0][2])

    def test_invalid_val_ratio(self):
        X = pd.DataFrame({"feature": range(10)})
        with pytest.raises(ValueError, match="val_ratio"):
            _split(X, train_ratio=0.5, val_ratio=0.0)

    def test_train_plus_val_exceeds_one(self):
        X = pd.DataFrame({"feature": range(10)})
        with pytest.raises(ValueError, match="train_ratio.*val_ratio"):
            _split(X, train_ratio=0.8, val_ratio=0.3)
