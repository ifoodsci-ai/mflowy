import pandas as pd


class FeatureImportanceMixin:
    @staticmethod
    def _build_importance_df(feature_names, importances) -> pd.DataFrame:
        """从 (names, importances) 序列对构造单行 DataFrame。

        各家 feature_names 属性名不同（feature_name_ / feature_names_ / feature_names_in_），
        但拿到值后构造逻辑一致。
        """
        return pd.DataFrame({n: [imp] for n, imp in zip(feature_names, importances)})
