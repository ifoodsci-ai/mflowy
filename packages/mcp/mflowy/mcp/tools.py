"""mflowy 平台 MCP 工具 — 词表查询 / mlflow 观测 / 工件读取 / 身份指纹 / 会话回收。

域工具（数据/建模）不住此层：域包自持 ``tools.py`` 并经 entry points 组
``mflowy.tools`` 注册，server/runner 发现即用（安装形态即通道——裸 host 只有
平台工具，装哪个域包多哪批工具）。本模块保持零域包导入：remote 系列平台工具
（ensure/logs/download，#153）消费 model 包 platform_client，**函数体内懒加载**
——注册无条件（entry points 静态），执行随安装形态（裸 host 无 model 时
fail-loud ImportError，不设降级机器，#159 §5）。

模块公有面即平台工具函数面（镜像契约钉在 tests/mcp/）；非工具 helper 必须
``_`` 前缀——server/runner 共用的注册辅助件 ``_tool_entry_points`` 即此形态。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from mflowy.utils import auth
from mflowy.utils.file import sha256_of
from pydantic import Field


def _tool_entry_points():
    """``mflowy.tools`` 组唯一消费口（server.register 与 runner 分发表共享）。

    撞名 fail-loud（红线 1）：SDK 撞名只警告且静默保第一个、dict 后写覆盖即
    静默遮蔽——两静默面在入口处统一拒绝，候选清单随报。
    """
    from importlib.metadata import entry_points

    eps = list(entry_points(group="mflowy.tools"))
    names = [ep.name for ep in eps]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise RuntimeError(f"mflowy.tools 组撞名（entry point 名须全仓唯一）: {dupes}")
    return eps


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
    """回收 POMDP 会话目录（clean/modeling 会话共用的生命周期基座；状态无关，按龄清理）。

    何时使用 sessions_gc：
    - 回收泄漏的弃置会话（活/死/损坏一律按 mtime 龄期，不读 manifest 内容）
    - 审计删除范围：先缺省 dry_run 看清单，确认后再 dry_run=false 执行
    """
    from mflowy.driver.session import SessionStore, gc_report_key

    if root is not None:
        roots = [Path(root)]
    else:
        base = Path.home() / ".mflowy"  # 平台约定：全部 POMDP 会话 root 住 ~/.mflowy/*_sessions
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


def _local_path(uri: str) -> str:
    """file:// URI 转本地绝对路径，远端 store 保持完整 URI（mlflow 查询工具共用约定）。"""
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    return str(Path(unquote(parsed.path))) if parsed.scheme == "file" else uri


def get_experiment(
    experiment_id: Annotated[str, Field(description="实验 ID（从任务输出 ExperimentInfo 获取）")],
    filter_steps: Annotated[
        list[str] | None,
        Field(description="可选，按节点 node_key 过滤（如 model__XGB；多节点合并取 top-N，不支持翻页）"),
    ] = None,
    max_results: Annotated[int, Field(description="每页最大数量")] = 10,
    page_token: Annotated[str | None, Field(description="翻页令牌（仅未过滤时有效）")] = None,
) -> dict:
    """实验仪表盘：实验本体（name/描述 tag/tags 含结构图/工件根/时间戳）+ run 状态目录。

    何时使用 get_experiment：
    - 从 experiment_id 浏览历史运行（可按节点过滤）——runs 目录：node/name/run_id/status/start_time
    - 获取 run_id 供 get_run/预测类工具引用
    - 查实验级 tags（workflow 指纹、mflowy.graph 结构图）与工件根
    详情（params/metrics/tags/artifacts）经 get_run 下钻。
    """
    client = _mlflow_client()
    exp = client.get_experiment(experiment_id)
    location = _local_path(exp.artifact_location)
    # run 状态目录（list_runs 吸收）：NodeResult 目录五元组投影 + start_time；
    # status 恒为 mlflow 原生（FINISHED/FAILED/RUNNING）——无引擎 skipped/"" 哨兵
    if filter_steps:
        # mlflow filter 不支持 tag IN / OR，按步骤等值查询后合并
        safe = [s.replace("'", "").strip() for s in filter_steps]
        merged = []
        for step in safe:
            merged.extend(
                client.search_runs(
                    experiment_ids=[experiment_id],
                    filter_string=f"tags.`mflowy.node` = '{step}'",
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
            page_token=page_token,
            order_by=["start_time DESC"],
        )
        runs, token = list(page), page.token
    run_dir = [
        {
            "node": r.data.tags.get("mflowy.node", ""),
            "name": r.info.run_name,
            "run_id": r.info.run_id,
            "status": r.info.status,
            "start_time": r.info.start_time,
        }
        for r in runs
    ]
    return {
        "experiment_id": exp.experiment_id,
        "name": exp.name,
        "description": exp.tags.get("mlflow.note.content", ""),
        "artifact_location": location,
        "lifecycle_stage": exp.lifecycle_stage,
        "creation_time": exp.creation_time,
        "last_update_time": exp.last_update_time,
        "tags": dict(exp.tags),
        "runs": run_dir,
        "next_page_token": token,
    }


def get_run(run_id: Annotated[str, Field(description="Run ID（从任务输出 RunInfo 或 get_experiment 获取）")]) -> dict:
    """查询单个 run 的详情：params/metrics/tags/artifacts 产物目录。

    何时使用 get_run：
    - 已有 run_id 时读取指标与参数
    - 查看 run 落盘的产物清单（图表/模型文件路径）；工件后端为本地文件树，文件本体可直接访问
    """
    client = _mlflow_client()
    run = client.get_run(run_id)
    d = run.to_dictionary()
    d["info"]["artifact_uri"] = run.info.artifact_uri
    root = run.info.artifact_uri
    d["artifacts"] = [
        {"path": _local_path(f"{root}/{a.path}"), "is_dir": a.is_dir, "file_size": a.file_size}
        for a in client.list_artifacts(run_id)
    ]
    return d


def login(
    base_url: Annotated[
        str | None,
        Field(description="平台地址；缺省读 MFLOWY_BASE_URL 环境变量，再缺省 http://localhost:30080"),
    ] = None,
    scopes: Annotated[
        list[str] | None,
        Field(description="申请的 scope 列表；缺省取服务端元数据 scope 目录全集（整体同意）"),
    ] = None,
    timeout_seconds: Annotated[int, Field(description="等待浏览器完成登录的时限秒数（缺省 300）")] = 300,
    force: Annotated[
        bool,
        Field(description="已登录时是否强制重新走浏览器授权（缺省 False：已登录直接返回状态）"),
    ] = False,
    device_name: Annotated[
        str | None,
        Field(description="设备显示名（平台 /account 已登录设备列表可见）；缺省 hostname (OS)"),
    ] = None,
) -> dict:
    """登录 mflowy 平台（OAuth 2.1：PKCE + 本机浏览器授权），凭据原子落盘 ~/.mflowy/credentials.json。

    何时使用 login：
    - 调用远程工具（训练/推理/部署）前建立或确认平台身份
    - 收到「需重新登录」提示后主动重新授权（force=true）
    - 检查当前登录状态（已登录则直接返回，不弹浏览器）
    """
    return auth.login(
        base_url=base_url,
        scopes=scopes,
        timeout_seconds=timeout_seconds,
        force=force,
        device_name=device_name,
    )


# ── remote 系列（#153）：平台实验就绪。client 详测住 model 包（#157），
# 本层只做编排；经 ``_platform_client`` 工厂缝替换假 client（#159 §8）。


def _platform_client():
    """平台 client 工厂缝（懒加载 model 包——见模块头「安装形态即通道」）。"""
    from mflowy.model.job_provider.platform_client import RemoteExperimentClient

    return RemoteExperimentClient()


def _md5_hex(path: Path) -> str:
    """本地指纹（#157 §4）：MD5 hex，与远端 etag 裸字符串全等比对。"""
    import hashlib

    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_file_manifest(files: list[str]) -> list[Path]:
    """清单预检（agent 显式声明，不做 YAML basename 推断）：全量 fail-loud 后再同步。"""
    from mflowy.utils.path import resolve_input_path

    resolved: list[Path] = []
    seen: dict[str, Path] = {}
    for raw in files:
        p = resolve_input_path(raw)  # 通道纪律：MCP 拒收相对路径（#159 §6）
        if p.name.startswith("."):
            raise ValueError(
                f"清单含 dotfile（{raw}）——上游 list 面不显示隐藏文件，逐次同步将判缺失死循环重传，整次拒绝"
            )
        if p.is_dir():
            raise ValueError(f"清单项是目录（{raw}）——只收文件（不做目录展开）")
        if not p.is_file():
            raise FileNotFoundError(f"清单项不存在：{raw}（解析为 {p}）")
        if p.name in seen:
            raise ValueError(
                f"清单内撞名：{seen[p.name]} 与 {p} 同 basename——远端名 /user-data/uploads/{p.name} 将互覆"
            )
        seen[p.name] = p
        resolved.append(p)
    return resolved


def _diff_entry(p: Path) -> dict:
    """同步 diff 条目单源（#156 §3 返回 shape：``{local, remote}``——size 不进
    返回面：不参与判定，agent 对本地文件自知 size，远端侧无需第二份复述）。"""
    return {"local": str(p), "remote": f"/user-data/uploads/{p.name}"}


def _sync_uploads(client, experiment_id: str, paths: list[Path]) -> dict:
    """增量同步（#156/#157）：查询 → 补缺 → 覆盖更新；不删除远端未列文件。

    指纹 = 本地 MD5 hex ↔ 远端 etag 裸字符串全等——「裸」= 剥 S3 引号后的串
    （网关/minio 恒带引号下发，不剥则永不相等、幂等面失效；S2 E2E 实证）；
    multipart etag（含 ``-``）与 MD5 hex 永不相等 → 自然落入重传分支（fail-safe）。
    """
    import mimetypes

    from mflowy.model.job_provider.platform_client import upload_presigned

    remote = {f.filename: f for f in client.list_files(experiment_id, "uploads")}
    result: dict[str, list[dict]] = {"uploaded": [], "updated": [], "skipped": []}
    for p in paths:
        entry = remote.get(p.name)
        if entry is not None and entry.etag.strip('"') == _md5_hex(p):
            result["skipped"].append(_diff_entry(p))
            continue
        content_type = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        url, headers = client.get_upload_url(experiment_id, "uploads", p.name, content_type)
        upload_presigned(url, headers=headers, source=p)
        result["updated" if entry is not None else "uploaded"].append(_diff_entry(p))
    return result


def _inheritable(client, experiment_id: str) -> list[dict]:
    """可继承面：outputs 根下存在 ``{job_id}.zip`` 的 job（status 不过滤——失败
    HPO 的部分 study db 可续跑；newest first；完整性校验在继承时做，#158 护栏）。"""
    entries = client.list_files(experiment_id, "outputs")
    job_ids = [e.filename[: -len(".zip")] for e in entries if e.filename.endswith(".zip") and "/" not in e.filename]
    jobs = [client.get_job(j) for j in job_ids]
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [{"job_id": j.id, "status": j.status} for j in jobs]


def ensure_platform_experiment(
    platform_experiment_id: Annotated[
        str | None,
        Field(
            description="平台实验 ID：传入=查询模式（校验存在，不存在 fail-loud，绝不静默转创建）；不传=创建模式（name 必填）"
        ),
    ] = None,
    files: Annotated[
        list[str] | None,
        Field(
            description="本地文件清单（绝对路径；MCP 通道拒收相对路径）——增量同步到 uploads："
            "MD5↔etag 全等跳过、不等覆盖更新；不删除远端未列文件"
        ),
    ] = None,
    name: Annotated[
        str | None, Field(description="新实验名（创建模式必填；与 platform_experiment_id 同传=参数来源冲突拒绝）")
    ] = None,
    desc: Annotated[str | None, Field(description="新实验描述（仅创建模式）")] = None,
) -> dict:
    """实验就绪（remote 系列前置动作）：查建平台实验、增量同步文件清单、回报可继承面。

    何时使用 ensure_platform_experiment：
    - 远程 compute 工具（hyperparameters_search / modeling / explanation / …）提交前：
      拿到 platform_experiment_id 并把本地数据/脚本同步到 uploads
    - 续跑场景：查该实验的可继承 zip 面（inheritable，newest first），取继承源 job_id
    - 重复调用幂等：同清单全部 skipped（MD5↔etag）；改动过的文件恰好落 updated

    返回 ``{platform_experiment_id, files: {uploaded/updated/skipped: [{local, remote}]},
    inheritable: [{job_id, status}]}``——远端名恒为 ``/user-data/uploads/<basename>``。
    """
    if platform_experiment_id is not None and (name is not None or desc is not None):
        raise ValueError(
            f"参数来源冲突：platform_experiment_id 与 name/desc 同传——查询模式不带创建参数（id={platform_experiment_id}）"
        )
    client = _platform_client()
    if platform_experiment_id is not None:
        experiment = client.get_experiment(platform_experiment_id)  # 404 → PlatformNotFoundError，绝不静默转创建
    elif name is None or not name.strip():
        raise ValueError("不传 platform_experiment_id 时 name 必填（创建模式）——两参全缺无从定位实验")
    else:
        experiment = client.create_experiment(name, desc or "")
    files_result: dict[str, list[dict]] = {"uploaded": [], "updated": [], "skipped": []}
    if files:
        files_result = _sync_uploads(client, experiment.id, _validate_file_manifest(files))
    return {
        "platform_experiment_id": experiment.id,
        "files": files_result,
        "inheritable": _inheritable(client, experiment.id),
    }
