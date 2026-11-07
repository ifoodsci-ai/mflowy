"""model 域 finding 分类学规格（#35 定案 / #120 词表单源 taxonomy.py）。

期望集是规格字面量（独立真源——诊断算子产出面与 LLM 落诊校验面的公共词表），
不从实现推导——防同义反复。
"""

from mflowy.model.taxonomy import KNOWN_FINDING_KINDS

_EXPECTED = frozenset(
    {
        "error_concentrated",
        "fold_std",
        "overfit_train_val_gap",
        "underfit_high_close",
        "overfit_val_loss_rise",
        "underfit_loss_plateau",
        "drift",
        "slice_collapse",
        "complexity_edge",
    }
)


def test_finding_taxonomy_matches_spec_literal():
    assert KNOWN_FINDING_KINDS == _EXPECTED
