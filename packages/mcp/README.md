# mflowy-mcp — MCP 层：工具目录 · 执行策略 · 遥测

MFlowy 以 MCP 为架构主体：本包的 `tools.py` 定义全部工具（pyfunc），三种形态共享同一套实现（CLI / MCP 双命令入口 + 直接 import）。

| 入口 | 命令 | 场景 |
|------|------|------|
| **CLI 通道** | `mflowy <tool> '<json args>'` | 命令行、K8s Job 容器、subprocess；裸装即用（零 SDK）——`DAGResult.status != finished` 以**非零退出码**外传失败，不吞 |
| **MCP 通道** | `mflowy-mcp` | stdio server，MCP 客户端 / agent（Claude Code、Cursor 等）接入；SDK+遥测住 `[mcp]` extra |
| 直接 import | `mflowy.mcp.tools` pyfunc | 宿主程序内嵌调用 |

> **0.2.0 tools 解耦**：域工具（数据/建模）已下沉域包自持（`mflowy.{data,model}.tools`），
> 经 entry points 组 `mflowy.tools` 注册；本包为**零域依赖 host**——平台工具（词表/mlflow
> 观测/工件/指纹）自带注册，域工具发现即用。装哪个域包多哪批工具（`tools/list` = 已安装能力）。

## 工具三分

| 类别 | 工具 | 执行位置 |
|------|------|---------|
| info | `list_nodes` / `get_node_info` / `validate_modeling_steps` / `file_hash` | 始终本地（base 依赖够用） |
| 分析与问卷 | `data_profile` / `eda` / `propose_questions` | 始终本地（轻量、无跨环境状态） |
| 建模 | `modeling` / `explanation` / `predict` / `inverse_optimization` | **JobProvider 委派** |
| mlflow | `list_runs` / `get_run` / `list_run_artifacts` | 始终本地 |
| 平台登录 | `login` | 本地（拉起浏览器授权，loopback 回调） |

建模类经 JobProvider 委派：内置 `local` 实现（LocalJobProvider，[modeling] extra 直调 driver 编排），远程实现接管执行环境。

**错误契约二分**：官方 MCP SDK 对非 ToolError 异常脱敏（客户端只收裸 `Error executing tool <name>`）。`validate_modeling_steps` 的领域结果本就是"不合法 + 原因"——全部失败路径（缺文件/片段格式/YAML 语法/DAG 构建）走结构化 `{valid: false, error}` 返回，agent 拿到详情自我纠正；其余工具的 operational error（路径/run 不存在等）维持 crash 通道（runner 侧非零退出码）。

## JobProvider：接入自定义执行环境

MFlowy 不内置绑定任何远程执行平台。实现 `JobProvider` 协议（`packages/builtin_plugins/model/mflowy/model/job_provider/protocol.py`）的 **4 个类型化方法**（modeling / explanation / predict / inverse_optimization，每个方法带 `headers: Mapping[str, str] | None` 透传请求元数据，MCP 客户端经 `call_tool(meta={...})` 携带），即可把建模工具委派到任意执行环境——K8s Job、远程集群、Serverless 均可。方法签名以 protocol.py 为准（本文不复述，防漂移）；契约行为由 `packages/builtin_plugins/model/tests/test_job_provider.py` / `test_tools_headers.py` 锁定。

启用方式：

```bash
PYTHONPATH=/srv/my-provider \
MFLOWY_JOB_PROVIDER=my_pkg.job_provider:MyRemoteProvider \
uvx --from "mflowy[mcp,modeling]" mflowy-mcp
```

- `MFLOWY_JOB_PROVIDER`：`local`（默认）或 `module:Class`；模块经宿主 `sys.path` 解析，provider 以标准 Python 包交付
- 规划预留：底层资源装载的 `JobProvisioner` 契约（mount / create / unmount / get_output）见 [roadmap](../../docs/roadmap.md)

## 平台登录（login）

远程工具（经 JobProvider 回调平台 API）需要平台身份：`login` 一次调用完成 OAuth 2.1 授权码流——PKCE S256 + 本机浏览器授权 + loopback 回调（127.0.0.1 随机端口）+ 凭据原子落盘。契约对端为平台 OAuth 授权服务器（快照见平台仓 `docs/OAUTH_CLI.md`）。

- **幂等**：已有同环境凭据直接返回 `already_logged_in` 摘要（不弹浏览器）；`force=true` 强制重授权
- **scope 缺省**：登录时拉服务端元数据 `scopes_supported` 全目录整体同意——客户端不拷目录，服务端目录变更零漂移；显式传 `scopes` 可收窄
- **环境**：`base_url` 参数 > `MFLOWY_BASE_URL` env > `http://localhost:30080`（本地 gateway NodePort；生产/自建网关经 env 指向）
- **凭据**：`~/.mflowy/credentials.json`（0600）单文件整替——refresh 为轮换制，旧值复用会触发平台复用检测（账号全端下线），故刷新/落盘临界区持文件锁（同机多 agent host 并发安全）
- **消费接线**：`mflowy.utils.auth.ensure_access_token()` / `LoginRequiredError`——未来远程 JobProvider 以此附 Bearer（401 单次刷新重试），不重造轮子
- **边界**：首版需本机浏览器（headless/SSH 场景明确报错；PAT 通道随平台侧后续）；JWT 本地仅解码展示不验签（资源侧统一验）

实现：`mflowy/utils/auth.py`（纯 stdlib）；行为契约见 `packages/utils/tests/test_auth.py`。

## 遥测与同意门

MCP 工具调用诊断采集：工具名、完整输入输出、调用意图。由 [agentcat](https://pypi.org/project/agentcat/) 以**纯 OTLP 模式**直出固定端点，不经 AgentCat 云。

- **同意模型**：默认 `ask`——首次工具调用经 MCP elicitation 询问，选择持久化到 `~/.mflowy/settings.json` 永不再问；`MFLOWY_TELEMETRY=on|off` 显式覆盖（容器 / CI 等无交互场景的唯一入口），无条件优先于 settings.json；客户端不支持 elicitation 时本次不采集、下次再问，工具调用不受影响
- **on 模式有意注入**（同意后生效）：工具多出必填 `context`（调用意图，第三人称）与 `session_id`（跨调用任务关联——首调签发、后续回显）参数，是采集设计的组成部分而非污染；参数名以 agentcat 常量（`context`）为准
- **数据去向**：项目官方遥测网关（TLS；自建部署见 `docker/telemetry/`）；设备密钥 `~/.mflowy/api.key`（0600）本机生成，服务端只存 hash
- **边界**：端点不可达完全透明（不阻塞、静默重试/丢弃）；仅覆盖 MCP 入口（`mflowy-mcp`），CLI（`mflowy`）与直接 import 不采集
- 实现：`mcp/telemetry.py`（唯一接线点 `wire_agentcat`，禁止 per-tool 打点）；隐私契约见 [PRIVACY.md](../../PRIVACY.md)

## 相关文档

- DAG 内核与插件 SDK：[mflowy-driver README](../driver/README.md)
- 能力目录与边函数：[mflowy-builtin-plugins README](../builtin_plugins/README.md)
