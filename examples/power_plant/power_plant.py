"""Combined Cycle Power Plant 数据导入（UCI 数据集 294）

三个入口对应链路各阶段：
- load              ：全量数据（EDA / 建模 / SHAP）
- load_X            ：仅特征列（逆向搜索 search_input）
- load_X_to_predict ：单行最优工况（上线前预测验证 predict）

数据获取：读本目录 ``power_plant.csv``（相对路径——与 notebook 运行约定
cwd=本目录一致）。python 节点安全扫描的导入白名单不含 pathlib/os/urllib，
执行环境也不注入 ``__file__``，脚本无法自定位/自下载；且 UCI 原始包为
zip+xlsx，pandas 2.3 的 read_excel URL-zip 通道已不可用。首次使用：从
UCI 294 下载 zip、解包 Folds5x2_pp.xlsx，按下方 rename 映射另存为本 csv。
"""

import pandas as pd


def _load_renamed() -> pd.DataFrame:
    df = pd.read_csv("power_plant.csv")
    return df.rename(
        columns={
            "AT": "temperature",
            "V": "exhaust_vacuum",
            "AP": "ambient_pressure",
            "RH": "relative_humidity",
            "PE": "net_hourly_electrical_energy_output",
        }
    )


def load() -> pd.DataFrame:
    """全量数据：4 个工况参数 + 净电能输出目标列。"""
    return _load_renamed()


def load_X() -> pd.DataFrame:
    """仅特征列（不含输出），供逆向搜索推断搜索空间。"""
    return _load_renamed().drop(columns=["net_hourly_electrical_energy_output"])


def load_X_to_predict() -> pd.DataFrame:
    """最优工况单行数据（search_input top-1，正式配置 300 trials），供上线前预测验证。"""
    return pd.DataFrame(
        [
            {
                "temperature": 3.25,
                "exhaust_vacuum": 28.05,
                "ambient_pressure": 1029.29,
                "relative_humidity": 80.18,
            }
        ]
    )
