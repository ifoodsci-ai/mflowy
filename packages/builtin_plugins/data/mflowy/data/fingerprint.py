"""数据面内容指纹（单一真源）：行哈希聚合——``index=False`` 免疫重放侧索引差。

消费方：cleaning_session（版本指纹 + 检测备忘键）、x_y 节点（``XySpec.fingerprint``）、
model split 契约（``SplitSpec.data_fingerprint`` 物化守卫）。

语义边界：

- **对行序/行标签不敏感**（设计选择：同内容集重放不误报）——守卫的是"值内容集"
  同一性，不是行序；跨 DataFrame 的多帧按参数序 ``|`` 连接
- 与文件级 sha256 tag（``utils.file.fingerprint_tags`` → ``mflowy.data_sha256``）是
  **两个粒度的血缘**：文件字节 vs DataFrame 行内容——load 后的解析/清洗/类型收敛
  改变行内容时只有本指纹能看见
- 同环境（同 pandas 版本）防错配守卫 + 血缘留痕，不承诺跨版本稳定
"""

from __future__ import annotations

import pandas as pd


def fingerprint(*dfs: pd.DataFrame) -> str:
    """帧内容指纹：各帧行哈希总和的 str，多帧 '|' 连接（X 与 y 分持时保序拼接）。

    零列帧（如 df 仅含目标列时的 X）无值可哈希——内容退化为行数，以 ``rows=n``
    记账（``hash_pandas_object`` 对 0 列帧返回零值对不齐行索引，直接调用会 raise）。
    """
    parts = [
        str(int(pd.util.hash_pandas_object(df, index=False).sum())) if df.shape[1] else f"rows={len(df)}" for df in dfs
    ]
    return "|".join(parts)
