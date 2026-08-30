# mflowy-mcp — MCP 层：工具目录 · 执行策略 · 遥测

MFlowy 以 MCP 为架构主体：本包的 `tools.py` 定义全部工具（pyfunc），三种入口共享同一套实现。

| 入口 | 命令 | 场景 |
|------|------|------|
| MCP server（stdio） | `mcpSrv` | MCP 客户端 / agent（Claude Code、Cursor 等）接入 |
| JSON runner（CLI） | `cmd <tool> '<json args>'` | 命令行、K8s Job 容器、subprocess |
| 直接 import | `mflowy.mcp.tools` pyfunc | 宿主程序内嵌调用 |

## 工具三分

| 类别 | 工具 | 执行位置 |
|------|------|---------|
| info | `list_modules` / `get_module_info` / `validate_modeling_steps` / `file_hash` | 始终本地（base 依赖够用） |
| 分析 | `data_profile` / `eda` / `infer_task_type_by_statistic` | 始终本地（轻量、无跨环境状态） |
| 建模 | `modeling` / `explanation` / `predict` / `inverse_optimization` | **JobProvider 委派** |
| mlflow | `list_runs` / `get_run` / `list_run_artifacts` | 始终本地 |

建模类经 JobProvider 委派：内置 `local` 实现（LocalJobProvider，[modeling] extra 直调 driver 编排），远程实现接管执行环境。

## JobProvider：接入自定义执行环境

MFlowy 不内置绑定任何远程执行平台。实现 `JobProvider` 协议（`mcp/job_provider/protocol.py`）的 **4 个类型化方法**（每个方法带 `headers: Mapping[str, str] | None` 透传请求元数据，MCP 客户端经 `call_tool(meta={...})` 携带），即可把建模工具委派到任意执行环境——K8s Job、远程集群、Serverless 均可。方法签名以 protocol.py 为准（本文不复述，防漂移）；契约行为由 `tests/mcp/test_job_provider.py` / `test_tools_headers.py` 锁定。

启用方式：

```bash
PYTHONPATH=/srv/my-provider \
MFLOWY_JOB_PROVIDER=my_pkg.job_provider:MyRemoteProvider \
uvx --from "mflowy[modeling]" mcpSrv
```

- `MFLOWY_JOB_PROVIDER`：`local`（默认）或 `module:Class`；模块经宿主 `sys.path` 解析，provider 以标准 Python 包交付
- 规划预留：底层资源装载的 `JobProvisioner` 契约（mount / create / unmount / get_output）见 [roadmap](../../docs/roadmap.md)

## 遥测与同意门

MCP 工具调用诊断采集：工具名、完整输入输出、调用意图。由 [agentcat](https://pypi.org/project/agentcat/) 以**纯 OTLP 模式**直出固定端点，不经 AgentCat 云。

- **同意模型**：默认 `ask`——首次工具调用经 MCP elicitation 询问，选择持久化到 `~/.mflowy/settings.json` 永不再问；`MFLOWY_TELEMETRY=on|off` 显式覆盖（容器 / CI 等无交互场景的唯一入口），无条件优先于 settings.json；客户端不支持 elicitation 时本次不采集、下次再问，工具调用不受影响
- **数据去向**：项目官方遥测网关（TLS；自建部署见 `docker/telemetry/`）；设备密钥 `~/.mflowy/api.key`（0600）本机生成，服务端只存 hash
- **边界**：端点不可达完全透明（不阻塞、静默重试/丢弃）；仅覆盖 MCP 入口（`mcpSrv`），`cmd` runner 与直接 import 不采集
- 实现：`mcp/telemetry.py`（唯一接线点 `wire_agentcat`，禁止 per-tool 打点）；隐私契约见 [PRIVACY.md](../../PRIVACY.md)

## 相关文档

- DAG 内核与插件 SDK：[mflowy-driver README](../driver/README.md)
- 能力目录与注入器：[mflowy-builtin-plugins README](../builtin_plugins/README.md)
