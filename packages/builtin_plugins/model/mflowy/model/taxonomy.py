"""建模域 finding 分类学（#35 定案词表）：诊断族产出面的闭合 kind 集。

诊断算子（diagnose：OOF 观察面 + 体征算子）与 holdout drift 的产出值域；LLM 落诊
（report_findings）的校验面。出题规则的订阅面随规则引擎退役（#120），词表单源归
此——轻量模块（工具 schema 描述引用本词表，工具面 import 期零重依赖）。
"""

KNOWN_FINDING_KINDS: frozenset[str] = frozenset(
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
