"""mflowy 平台 MCP 工具 — 词表查询 / mlflow 观测 / 工件读取 / 身份指纹 / 会话回收。

域工具（数据/建模）不住此层：域包自持 ``tools.py`` 并经 entry points 组
``mflowy.tools`` 注册，server/runner 发现即用（安装形态即通道——裸 host 只有
平台工具，装哪个域包多哪批工具）。本模块保持零域包导入。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from mflowy.utils.file import sha256_of
from pydantic import Field


def file_hash(
    path: Annotated[str, Field(description="文件绝对路径")],
) -> dict:
    """为文件当前内容生成稳定指纹。

    何时使用 file_hash：
    - 上传文件(如CSV)身份核验
    - 跨阶段篡改检测
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"错误: 文件不存在: {path}")
    if not file_path.is_file():
        raise ValueError(f"错误: 路径不是文件: {path}")

    digest = sha256_of(file_path)
    size = file_path.stat().st_size
    return {
        "path": path,
        "sha256": digest,
        "size_bytes": size,
    }


def _requires(namespace: str, node_name: str) -> str | None:
    """缺依赖标注：域包 extras 的可用性声明（知识随域居住，host 惰性取用）。

    0.2.0 唯一可选 extra = model 包 ``[deep]``（torch）；其余能力为域包硬依赖零标注。
    域包未安装时其节点本就不在目录（词表=已安装能力），无需标注。
    """
    from mflowy.driver.config import node_key
    from mflowy.driver.discover import GROUPS, entry

    ep = entry(node_key(namespace, node_name))
    if ep is None or ep.group != GROUPS[0]:
        return None
    mod = ep.value.split(":")[0]
    if mod.startswith("mflowy.model."):
        import importlib

        try:
            extras = importlib.import_module("mflowy.model.extras")
        except ImportError:
            return None
        return extras.extra_of(ep.value)
    return None


def sessions_gc(
    root: Annotated[
        str | None,
        Field(description="显式会话根目录（env 自定义 root 时指定；缺省扫描 ~/.mflowy/*_sessions）"),
    ] = None,
    ttl_days: Annotated[int, Field(description="过期天数（mtime 锚 manifest.json；缺省 30）")] = 30,
    dry_run: Annotated[bool, Field(description="只列不删（缺省 True——agent 手里的删除工具安全缺省）")] = True,
) -> dict:
    """回收 MDP 会话目录（clean/modeling 会话共用的生命周期基座；状态无关，按龄清理）。

    何时使用 sessions_gc：
    - 回收泄漏的弃置会话（活/死/损坏一律按 mtime 龄期，不读 manifest 内容）
    - 审计删除范围：先缺省 dry_run 看清单，确认后再 dry_run=false 执行
    """
    from mflowy.driver.mdp_session import SessionStore, gc_report_key

    if root is not None:
        roots = [Path(root)]
    else:
        base = Path.home() / ".mflowy"  # 平台约定：全部 MDP 会话 root 住 ~/.mflowy/*_sessions
        roots = sorted(p for p in base.glob("*_sessions") if p.is_dir())
    action_key = gc_report_key(dry_run)
    report: dict[str, list] = {"scanned_roots": [str(r) for r in roots], action_key: [], "ignored": []}
    for r in roots:
        part = SessionStore(r).gc(ttl_days=ttl_days, dry_run=dry_run)
        report[action_key].extend(part[action_key])
        report["ignored"].extend(part["ignored"])
    return report


def list_nodes(
    namespace: Annotated[
        str | None,
        Field(description="可选，过滤 namespace（load/clean/x_transformer/model/plot/statistic；非归组节点在空组）"),
    ] = None,
) -> list[dict]:
    """查询 mflowy 已注册的节点，按 namespace 分组。

    何时使用 list_nodes：
    - 编写 modeling YAML 前盘点可用模块
    - 确认某 namespace 下有哪些节点可选
    - 查看 requires 标注——当前环境缺 extra 时先装（如 `uv pip install "mflowy[modeling]"`）

    缺 extra 依赖的节点降级为 ``bindings=None`` 条目（不可内省）并附 requires
    标注——装好 extra 再查即得参数边依赖。
    """
    from mflowy.driver.module import list_nodes as _list

    result = []
    for s in _list(namespace):
        d = asdict(s)
        # requires 以裸节点名对键（NodeEntry.name 存 node_key——经 config 单源逆函数还原）
        from mflowy.driver.config import split_node_key

        requires = {
            bare: r for e in d["nodes"] if (r := _requires(d["namespace"], (bare := split_node_key(e["name"])[1])))
        }
        if requires:
            d["requires"] = requires
        result.append(d)
    return result


def get_node_info(
    namespace: Annotated[str, Field(description="归组（如 model）")],
    node_name: Annotated[str, Field(description="节点名（如 XGB）")],
) -> dict:
    """查询指定节点的 YAML 配置模板(参数/默认值/参数边依赖)。

    何时使用 get_node_info：
    - 选定模块后查看参数与 YAML 写法
    - 校对参数拼写、类型、默认值
    - 当前环境缺 extra 依赖时返回 available=false 与所需 extra（参数详情不可得）
    """
    from mflowy.driver.discover import CatalogMiss
    from mflowy.driver.module import get_node_info as _get

    requires = _requires(namespace, node_name)
    try:
        info = asdict(_get(namespace, node_name))
    except CatalogMiss:
        raise  # 词表 miss（节点名打错）：带 Available 列表原音透传，不是「缺 extra」
    except ImportError as e:
        if requires is None:
            raise  # 与 extra 无关的导入失败，维持原报错
        return {
            "name": f"{namespace}.{node_name}",
            "namespace": namespace,
            "node_name": node_name,
            "description": "",
            "parameters_schema": {"type": "object", "properties": {}},
            "requires": requires,
            "available": False,
            "reason": f'当前环境缺少 [{requires}] extra 依赖（{e}）；安装如：uv pip install "mflowy[{requires}]"',
        }
    if requires:
        info["requires"] = requires
    return info


# ── mlflow 工具（始终本地执行，base 含 mlflow SDK） ────────────────────────

_client = None


def _mlflow_client():
    from mflowy.utils.mlflow import setup

    global _client
    if _client is not None:
        setup()  # hot reload tracking_uri from $MLFLOW_TRACKING_URI
        return _client

    setup()
    from mlflow.tracking import MlflowClient

    _client = MlflowClient()
    return _client


def list_runs(
    experiment_id: Annotated[str, Field(description="实验 ID（从任务输出 ExperimentInfo 获取）")],
    filter_steps: Annotated[
        list[str] | None,
        Field(description="可选，按节点 node_key 过滤（如 model__XGB；多节点合并取 top-N，不支持翻页）"),
    ] = None,
    max_results: Annotated[int, Field(description="每页最大数量")] = 10,
    page_token: Annotated[str | None, Field(description="翻页令牌（仅未过滤时有效）")] = None,
) -> dict:
    """批量查询实验下所有节点的 run 信息。

    何时使用 list_runs：
    - 从 experiment_id 浏览历史运行(可按节点过滤)
    - 获取 run_id 供 get_run/预测类工具引用
    """
    client = _mlflow_client()
    if filter_steps:
        # mlflow filter 不支持 tag IN / OR，按步骤等值查询后合并
        safe = [s.replace("'", "").strip() for s in filter_steps]
        merged = []
        for s in safe:
            merged.extend(
                client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.`mflowy.node` = '{s}'",
                    max_results=max_results,
                    order_by=["start_time DESC"],
                )
            )
        merged.sort(key=lambda r: r.info.start_time, reverse=True)
        runs, token = merged[:max_results], None
    else:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            max_results=max_results,
            order_by=["start_time DESC"],
            page_token=page_token,
        )
        runs, token = list(page), page.token
    result = []
    for r in runs:
        d = r.to_dictionary()
        d["info"]["artifact_uri"] = r.info.artifact_uri
        result.append(d)
    return {"runs": result, "next_page_token": token}


def get_run(run_id: Annotated[str, Field(description="Run ID（从任务输出 RunInfo 或 list_runs 获取）")]) -> dict:
    """查询单个节点的详细信息(metrics/params/tags/artifact_uri)。

    何时使用 get_run：
    - 已有 run_id 时读取指标与参数
    """
    client = _mlflow_client()
    run = client.get_run(run_id)
    d = run.to_dictionary()
    d["info"]["artifact_uri"] = run.info.artifact_uri
    return d


def list_run_artifacts(
    run_id: Annotated[str, Field(description="Run ID")],
    path: Annotated[str | None, Field(description="可选，artifact 子路径过滤")] = None,
) -> list[dict]:
    """查询执行步骤的文件产物列表。

    何时使用 list_run_artifacts：
    - 查看 run 落盘的图表/模型文件
    - 定位产物路径供用户查看
    """
    from urllib.parse import unquote, urlparse

    client = _mlflow_client()
    artifacts = client.list_artifacts(run_id, path)
    # FileInfo.path 相对 run 的 artifact_uri；拼全后 file:// root 转本地绝对路径，远端 store 保持完整 URI
    root = client.get_run(run_id).info.artifact_uri
    result = []
    for a in artifacts:
        uri = f"{root}/{a.path}"
        parsed = urlparse(uri)
        loc = str(Path(unquote(parsed.path))) if parsed.scheme == "file" else uri
        result.append({"path": loc, "is_dir": a.is_dir, "file_size": a.file_size})
    return result


def read_artifact(
    run_id: Annotated[str, Field(description="Run ID")],
    artifact_path: Annotated[
        str, Field(description="artifact 相对路径（如 statistics.parquet，list_run_artifacts 可查）")
    ],
    columns: Annotated[list[str] | None, Field(description="列投影（parquet/csv），None 全列")] = None,
    rows: Annotated[
        str, Field(description='行选择："head:N" 前 N 行 / "sample:N" 随机 N 行，默认 head:20')
    ] = "head:20",
) -> dict:
    """读取 run 产物为 token 友好摘要（agent 的钻取/审计兜底通道）。

    何时使用 read_artifact：
    - 画像表/双变量长表摘要不够看，需要核对具体行/列证据
    - 审计历史 run 的清洗报告/统计表

    parquet/csv 按列投影 + 行限返回 markdown 渲染与总形状元数据（返回单元格
    超上限自动截断并标记）；json/txt 文本透传（大小封顶）；png/模型等二进制只
    返回元数据不渲染。主食应是会话观测（s′ 报告）与统计表摘要，本工具是兜底。
    """
    from urllib.parse import unquote, urlparse

    client = _mlflow_client()
    root = client.get_run(run_id).info.artifact_uri
    uri = f"{root}/{artifact_path}"
    parsed = urlparse(uri)
    path = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(uri)
    if not path.exists():
        raise FileNotFoundError(f"产物不存在: {artifact_path}（run {run_id}，可用 list_run_artifacts 查清单）")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".csv"}:
        import pandas as pd

        df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path)
        shape = list(df.shape)
        if columns:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"列不存在: {missing}，可用列: {df.columns.tolist()}")
            df = df[columns]
        n = _parse_rows(rows, default=20)
        df = df.head(n) if rows.startswith("head") else df.sample(min(n, len(df)), random_state=0)
        cap = 2000  # 返回单元格上限（token 护栏）
        truncated = False
        if df.shape[0] * df.shape[1] > cap:
            keep = max(1, cap // df.shape[1])
            df = df.head(keep)
            truncated = True
        rendered = _md_table(df)
        return {
            "kind": suffix.lstrip("."),
            "path": str(path),
            "shape": shape,
            "returned": list(df.shape),
            "truncated": truncated,
            "rendered": rendered,
        }
    if suffix in {".json", ".txt", ".md", ".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        cap = 8000
        return {
            "kind": suffix.lstrip("."),
            "path": str(path),
            "truncated": len(text) > cap,
            "rendered": text[:cap],
        }
    return {
        "kind": "binary",
        "path": str(path),
        "file_size": path.stat().st_size,
        "note": "二进制产物不渲染；图表请引用路径，模型经 model.loader 消费",
    }


def _parse_rows(rows: str, default: int) -> int:
    try:
        return int(rows.split(":")[1])
    except (IndexError, ValueError):
        return default


def _md_table(df, cell_cap: int = 40) -> str:
    """极简 markdown 表渲染（不引 tabulate 依赖）；长单元格截断防单行爆 token。"""

    def cell(v) -> str:
        if v is None or (isinstance(v, float) and v != v):  # None/NaN
            return ""
        s = str(v)
        return s if len(s) <= cell_cap else s[: cell_cap - 1] + "…"

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([" --- "] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)
