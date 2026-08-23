"""Combined Cycle Power Plant 数据导入（UCI 数据集 294）

三个入口对应链路各阶段：
- load              ：全量数据（EDA / 建模 / SHAP）
- load_X            ：仅特征列（逆向搜索 search_input）
- load_X_to_predict ：单行最优工况（上线前预测验证 predict）

数据获取：首次调用自动从 UCI 拉取 zip+xlsx 并落盘 power_plant.csv（幂等，
已存在则跳过），不依赖 notebook 前置准备。路径基于 Path(__file__) 解析（与进程 CWD 解耦）。
"""

from pathlib import Path

import pandas as pd


def _ensure_data() -> None:
    """幂等下载：power_plant.csv 不存在时从 UCI 拉取（zip+xlsx 由 pandas 原生 URL 读取，
    原始列名 AT/V/AP/RH/PE 一次 rename；不经 requests/zipfile，无需扩展 scan 白名单）。"""
    data_file = Path(__file__).resolve() / "power_plant.csv"
    if data_file.exists():
        return
    df = pd.read_excel("https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip")
    df = df.rename(
        columns={
            "AT": "temperature",
            "V": "exhaust_vacuum",
            "AP": "ambient_pressure",
            "RH": "relative_humidity",
            "PE": "net_hourly_electrical_energy_output",
        }
    )
    df.to_csv(data_file, index=False)


def load() -> pd.DataFrame:
    """全量数据：4 个工况参数 + 净电能输出目标列。"""
    _ensure_data()
    return pd.read_csv(Path(__file__).resolve() / "power_plant.csv")


def load_X() -> pd.DataFrame:
    """仅特征列（不含输出），供逆向搜索推断搜索空间。"""
    _ensure_data()
    return pd.read_csv(Path(__file__).resolve() / "power_plant.csv").drop(
        columns=["net_hourly_electrical_energy_output"]
    )


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
