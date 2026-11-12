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


# 远程镜像契约字面常量（#159 §4：**非 env 读取**——本地同名 env 属 MCP 路由缺省语义，
# 与远端镜像烧 ENV 契约无关；镜像面见 docker/Dockerfile 烧 ENV、容器面见 jobwrap）。
# 观察面单参推导（#160 终稿）：工件 URI 命中此前缀 → 改写为 tracking db 同根本地路径。
REMOTE_TASK_DIR = "/user-data/workspace"


def _rewrite_remote_artifact(value: str, local_root: Path | None) -> str:
    """远程工件路径改写（get_experiment/get_run 共用纯函数）。

    命中 ``REMOTE_TASK_DIR`` 前缀（file:// URI 或裸路径、精确或子路径）→ 替换为
    ``local_root``（= tracking db 所在目录——zip 契约保证 mlflow.db 与 ``.mlruns/``
    同根，db 目录即工件根，#155）；不匹配或 ``local_root`` 缺席（非 sqlite 库/本地
    缺省查询）原样透传（Default 实验路径实证从未写入远端前缀）。
    """
    if local_root is None:
        return value
    stripped = _local_path(value)
    marker = REMOTE_TASK_DIR
    if stripped == marker:
        return str(local_root)
    if stripped.startswith(marker + "/"):
        return str(local_root) + stripped[len(marker) :]
    return value


def _tracking_client(tracking_uri: str | None):
    """观察面 client 解析：缺省 = 机器默认库（全局 _mlflow_client）；给定 = 逐调用
    构造（零全局态，#159 §4），并推导工件根改写（sqlite 库文件所在目录）。"""
    if tracking_uri is None:
        return _mlflow_client(), None
    from urllib.parse import unquote, urlparse

    from mlflow.tracking import MlflowClient

    parsed = urlparse(tracking_uri)
    if parsed.scheme == "sqlite":
        # sqlite:////abs（四斜杠绝对路径）经 urlparse 出 "//x" 形态——前导斜杠归一
        db_path = Path("/" + unquote(parsed.path).lstrip("/"))
        return MlflowClient(tracking_uri=tracking_uri), db_path.parent
    return MlflowClient(tracking_uri=tracking_uri), None  # http 等远程库：无同根推导，透传


def get_experiment(
    experiment_id: Annotated[str, Field(description="实验 ID（从任务输出 ExperimentInfo 获取）")],
    filter_steps: Annotated[
        list[str] | None,
        Field(description="可选，按节点 node_key 过滤（如 model__XGB；多节点合并取 top-N，不支持翻页）"),
    ] = None,
    max_results: Annotated[int, Field(description="每页最大数量")] = 10,
    page_token: Annotated[str | None, Field(description="翻页令牌（仅未过滤时有效）")] = None,
    tracking_uri: Annotated[
        str | None,
        Field(
            description="可选，逐调用追踪库 URI（如 sqlite:///<download 目录>/{job_id}/mlflow.db）——远程任务产物下载后本地查询用；工件路径随之改写为同根本地路径"
        ),
    ] = None,
) -> dict:
    """实验仪表盘：实验本体（name/描述 tag/tags 含结构图/工件根/时间戳）+ run 状态目录。

    何时使用 get_experiment：
    - 从 experiment_id 浏览历史运行（可按节点过滤）——runs 目录：node/name/run_id/status/start_time
    - 获取 run_id 供 get_run/预测类工具引用
    - 查实验级 tags（workflow 指纹、mflowy.graph 结构图）与工件根
    详情（params/metrics/tags/artifacts）经 get_run 下钻。
    """
    client, artifact_root = _tracking_client(tracking_uri)
    exp = client.get_experiment(experiment_id)
    location = _rewrite_remote_artifact(_local_path(exp.artifact_location), artifact_root)
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


def get_run(
    run_id: Annotated[str, Field(description="Run ID（从任务输出 RunInfo 或 get_experiment 获取）")],
    tracking_uri: Annotated[
        str | None,
        Field(
            description="可选，逐调用追踪库 URI（如 sqlite:///<download 目录>/{job_id}/mlflow.db）——远程任务产物下载后本地查询用；工件路径随之改写为同根本地路径"
        ),
    ] = None,
) -> dict:
    """查询单个 run 的详情：params/metrics/tags/artifacts 产物目录。

    何时使用 get_run：
    - 已有 run_id 时读取指标与参数
    - 查看 run 落盘的产物清单（图表/模型文件路径）；工件后端为本地文件树，文件本体可直接访问
    - 远程任务产物：download_platform_job_artifacts 拉取后传 tracking_uri
      （``sqlite:///{target_dir}/{job_id}/mlflow.db``）查该继承链的 run——工件路径
      自动从 ``/user-data/workspace`` 改写为解压根的本地路径（run 工件与 logged
      models 双形态均覆盖）
    """
    client, artifact_root = _tracking_client(tracking_uri)
    run = client.get_run(run_id)
    d = run.to_dictionary()
    d["info"]["artifact_uri"] = _rewrite_remote_artifact(run.info.artifact_uri or "", artifact_root)
    root = run.info.artifact_uri
    d["artifacts"] = [
        {
            "path": _rewrite_remote_artifact(_local_path(f"{root or ''}{a.path}"), artifact_root),
            "is_dir": a.is_dir,
            "file_size": a.file_size,
        }
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


def get_platform_job_logs(
    job_id: Annotated[
        str, Field(description="平台任务 ID（从 compute 工具的 JobOutcome.submitted 或 ensure 的 inheritable 取得）")
    ],
    tail_lines: Annotated[int, Field(description="日志尾行数（1–10000；默认 500）——越界本地拒绝，不发出请求")] = 500,
) -> dict:
    """查平台任务状态与日志尾（stateless 轮询——agent 唯一在途状态面）。

    何时使用 get_platform_job_logs：
    - compute 工具返回 JobOutcome(submitted) 后轮询任务进度：``terminal=true`` 即停
      （终态 = succeeded | failed | lost | canceled），否则继续轮询
    - 任务失败时取日志尾定位（zip 拉取走 download_platform_job_artifacts）

    返回 ``{job_id, status, terminal, tail_lines, logs}``——status/terminal/logs
    合并返回（一次调用即全貌）；``tail_lines`` 上游容错（<=0→500）非我方语义，
    本工具 1–10000 fail-loud。pod 未调度期上游 logs 端点 404（"job pods not
    found yet"）——job 在场（get_job 通过）即瞬态，logs 取空串继续轮询；job
    本身不存在仍 404 fail-loud。
    """
    if not 1 <= tail_lines <= 10000:
        raise ValueError(f"tail_lines 须在 1–10000（got {tail_lines}）——上游 <=0→500 是其容错非本工具语义")
    from mflowy.model.job_provider.platform_client import PlatformNotFoundError

    client = _platform_client()
    job = client.get_job(job_id)  # job 不存在 → 404 fail-loud
    try:
        logs = client.get_job_logs(job_id, tail_lines)
    except PlatformNotFoundError:
        logs = ""  # 瞬态：pod 未调度（S4 实测）——合并视图继续轮询
    from mflowy.model.job_provider.platform_client import JOB_TERMINAL_STATUSES

    return {
        "job_id": job_id,
        "status": job.status,
        "terminal": job.status in JOB_TERMINAL_STATUSES,
        "tail_lines": tail_lines,
        "logs": logs,
    }


def download_platform_job_artifacts(
    job_id: Annotated[str, Field(description="平台任务 ID（JobOutcome.submitted 的 job_id 或 ensure inheritable 面）")],
    platform_experiment_id: Annotated[str, Field(description="平台实验 ID（任务 zip 所在实验——outputs 面寻址）")],
    target_dir: Annotated[
        str,
        Field(
            description="解压根目录（须已存在；MCP 通道须绝对路径）——解压进 target_dir/{job_id}/，原子替换可重复调用"
        ),
    ],
) -> dict:
    """拉取平台任务产物：下载 outputs 下 ``{job_id}.zip`` → 解压进 ``target_dir/{job_id}/``（zip 根 = workspace 根）。

    何时使用 download_platform_job_artifacts：
    - get_platform_job_logs 轮询到终态后拉取产物（重下载 = 原子替换，幂等）
    - 拉取后本地查询：get_run/get_experiment 传 ``tracking_uri=sqlite:///{target_dir}/{job_id}/mlflow.db``
      （mlflow.db 与 .mlruns 同根——db 目录即工件根，观察面自动改写为本地路径）

    zip 本体即弃（不落 target_dir）；子目录 ``{job_id}/`` 是本工具自有领地。
    返回 ``{job_id, platform_experiment_id, local_task_dir, file_count, mlflow_db}``。
    """
    import shutil
    import tempfile
    import zipfile

    from mflowy.utils.path import resolve_input_path

    target = resolve_input_path(target_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"target_dir 不存在或非目录：{target_dir}（解析为 {target}）——先创建再调用")
    client = _platform_client()
    url, _meta = client.get_download_url(platform_experiment_id, "outputs", f"{job_id}.zip")
    from mflowy.model.job_provider.platform_client import download_presigned

    with tempfile.TemporaryDirectory(prefix="mflowy_download_") as td:
        zip_path = Path(td) / f"{job_id}.zip"
        download_presigned(url, zip_path)  # zip 本体即弃——临时目录退场即删
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                staged = Path(td) / "unpacked"
                zf.extractall(staged)
        except zipfile.BadZipFile as e:
            raise ValueError(f"平台产物 zip 损坏（job={job_id}）：{e}——上游 outputs 面异常，重试或报平台") from e
        dest = target / job_id
        replacing = target / f".{job_id}.replacing"
        retired = target / f".{job_id}.retired"
        shutil.rmtree(retired, ignore_errors=True)  # 上轮残留即清
        staged.rename(replacing)  # 同盘 rename：tmp 解压 → 换位
        if dest.exists():
            dest.rename(retired)
        replacing.rename(dest)
        shutil.rmtree(retired, ignore_errors=True)
    mlflow_db = dest / "mlflow.db"
    return {
        "job_id": job_id,
        "platform_experiment_id": platform_experiment_id,
        "local_task_dir": str(dest),
        "file_count": len(names),
        "mlflow_db": str(mlflow_db) if mlflow_db.is_file() else None,
    }
