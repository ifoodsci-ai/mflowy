"""数据面内容指纹（单一真源）：CAS——df 规范序列化（``to_parquet(index=False)``）字节流的 sha256。

消费方：cleaning_session 的 ``_detect_all`` 备忘键（纯内存优化设施——manifest
不落指纹键，版本溯源由 artifact_sha256 单键承担，重放对账经闭合性直对工件 sha）。
model 域经 #66 划分引用化后不再消费——划分同一性由 split.pkl 副产物的 run_id
引用承担，无指纹重现假设。

语义边界：

- **CAS 闭合**：与 ``to_parquet(index=False)`` 落盘工件同参序列化 → 指纹 = 工件
  sha256——重放/审计场景内存 df 指纹可直接对盘上工件 sha 对账（与文件级血缘
  ``mflowy.data_sha256`` 在规范字节表示上同一粒度）
- **行序进指纹**（CAS：行序是字节内容的一部分——重排 = 不同数据）；**行标签不进**
  （``index=False``——重放侧索引差不误报）
- 零列帧 parquet 不记行数——内容退化为 ``rows=n`` 记账后再哈希（格式统一 64 位 hex）
- 同环境（同 pandas/pyarrow 版本）防错配守卫 + 血缘留痕，不承诺跨版本稳定
  （parquet footer 携带创建器版本）
"""

from __future__ import annotations

import hashlib
import io

import pandas as pd


def fingerprint(df: pd.DataFrame) -> str:
    """帧内容地址：规范序列化（``to_parquet(index=False)``）字节流的 sha256 十六进制。

    零列帧（如 df 仅含目标列时的 X）parquet 无列可写、行数不进字节——内容退化为
    ``rows=n`` 记账后再哈希。
    """
    if not df.shape[1]:
        return hashlib.sha256(f"rows={len(df)}".encode()).hexdigest()
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return hashlib.sha256(buf.getvalue()).hexdigest()
