# mflowy-mcp — MCP 层：入口 · 工具目录 · 平台登录 · 遥测

MFlowy 以 MCP 为架构主体：全部能力以工具（pyfunc）暴露，经 entry points 组 `mflowy.tools` 发现——域工具住各自域包、平台工具住本包（见下方「0.2.0 tools 解耦」），三种形态共享同一套实现（CLI / MCP 双命令入口 + 直接 import）。

| 入口 | 命令 | 场景 |
|------|------|------|
| **CLI 通道** | `mflowy <tool> '<json args>'` | 命令行、K8s Job 容器、subprocess；裸装即用（零 SDK）——`DAGResult.status != finished` 以**非零退出码**外传失败，不吞 |
| **MCP 通道** | `mflowy-mcp` | stdio server，MCP 客户端 / agent（Claude Code、Cursor 等）接入；SDK+遥测住 `[mcp]` extra |
| 直接 import | `mflowy.{mcp,data,model}.tools` pyfunc | 宿主程序内嵌调用 |
| **容器入口** | `mflowy-job wrap --tool <t> --kwargs-json <j> [--inherit-from <job>]` | 平台 Job 容器（#158：继承解包 → 执行 → 无条件 zip 归档，工具 rc 透传） |

**工件锚定（task_dir）双路由**（#95）：两入口路由到同一工具函数，锚由入口中间件（`route_task_dir`）决定——工具以 `@anchor_task_dir("<参数名>")` 声明锚参数（纯元数据，签名/schema 零影响），从业务路径参数推父目录为任务目录（实验工件根 = `task_dir/.mlruns`）。CLI：`MFLOWY_TASK_DIR` env **钉住**（推导不覆写）→ 未设时推导 → 缺省 `cwd`；MCP：推导 → 缺省 `~/.mflowy`（server 无业务 CWD 语义）。MCP 侧数据路径须绝对（相对路径不推锚，解析处 fail-loud）。

> **0.2.0 tools 解耦**：域工具（数据/建模）已下沉域包自持（`mflowy.{data,model}.tools`），
> 经 entry points 组 `mflowy.tools` 注册；本包为**零域依赖 host**——平台工具（词表/mlflow
> 观测/工件/指纹/登录）由 mflowy-mcp pyproject **同组静态声明**（与域包同构，server/runner
> 同一真源消费，镜像契约钉在 `tests/mcp/test_platform_tools_registry.py`），域工具发现即用。
> remote 系列平台工具（`ensure_platform_experiment` 等）的引擎住 model 包 platform_client，**函数体内懒加载**——注册无条件、执行随安装形态（裸 host 缺 model 时调用 fail-loud ImportError；`mflowy[mcp]` 聚合 extra 已含 model）。
> 装哪个域包多哪批工具（`tools/list` = 已安装能力）。

## 工具分组（执行位置）

| 类别 | 工具 | 执行位置 |
|------|------|---------|
| info | `list_nodes` / `get_node_info` / `file_hash` | 始终本地（base 依赖够用） |
| 分析与问卷 | `data_profile` / `eda` / `propose_questions` | 始终本地（轻量、无跨环境状态） |
| POMDP 会话 | `clean_*` 六操作 + `modeling_*` 七操作（含 `report_findings`） | 始终本地（会话态在本地盘上工件） |
| 报告素材 | `draw_taylor_diagram` / `modeling_report_skeleton` | 始终本地 |
| YAML 预检 | `validate_modeling_steps` | 始终本地（结构化 `{valid: false, error}` 错误契约） |
| 建模 compute | `modeling` / `hyperparameters_search` / `explanation` / `predict` / `inverse_optimization`（受理面 = JobOutcome：finished 内嵌 DAGResult / submitted 提交凭据） | **JobProvider 委派**（local / remote） |
| 执行模式 | `switch_job_provider`（model 包，local↔remote 进程级切换） | 本地 |
| mlflow 观测 | `get_experiment` / `get_run` | 始终本地（工件后端为本地文件树） |
| 平台 | `login`（OAuth 2.1 浏览器授权）/ `sessions_gc`（TTL 回收） | 本地 |
| 平台·remote | `ensure_platform_experiment`（实验就绪：查建 + 文件增量同步 + 可继承面） | 平台 API（经 model 包 platform_client，懒加载） |
| 平台·remote | `get_platform_job_logs`（stateless 轮询：status/terminal/logs 合并返回） | 平台 API（经 model 包 platform_client，懒加载） |

建模类经 JobProvider 委派（内置 `local` / 自定义 `module:Class`）——协议与自定义执行环境接入见 [mflowy-model README](../builtin_plugins/model/README.md)「JobProvider」节（预测模型训练子系统）。

**错误契约**：结构化失败面（`validate_modeling_steps` 的 `{valid: false, error}`）之外，工具的 fail-loud 报错（红线 1 的候选清单式文案——路径/run 不存在、参数冲突等）在 MCP 通道经 `server._fail_loud` 统一转 `ToolError` 透传（消息带异常类名；SDK 的多租户脱敏缺省对本地工具无益，CLI/runner 通道本就全量打印 + 非零退出码）；crash 的完整 traceback 仍在 server stderr。

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

MCP 工具调用诊断采集：工具名、完整输入输出、调用意图。由 [agentcat](https://pypi.org/project/agentcat/) 以**纯 OTLP 模式**直出配置端点（默认官方网关），不经 AgentCat 云。

- **同意模型**：默认 `ask`——首次工具调用经 MCP elicitation 询问，选择持久化到 `~/.mflowy/settings.json` 永不再问；`MFLOWY_TELEMETRY=on|off` 显式覆盖（容器 / CI 等无交互场景的唯一入口），无条件优先于 settings.json；客户端不支持 elicitation 时本次不采集、下次再问，工具调用不受影响
- **on 模式有意注入**（同意后生效）：工具多出必填 `context`（调用意图，第三人称）与 `session_id`（跨调用任务关联——首调签发、后续回显）参数，是采集设计的组成部分而非污染；参数名以 agentcat 常量（`context`）为准
- **遥测副本脱敏**（#189，双钩子随 `wire_agentcat` 接线，agentcat 保证事件级先于字符串级、均 fail-closed）：事件级 `redact_event` 按字段规则表 `TELEMETRY_FIELD_RULES` 剔除标识字段（表内容以常量为源，表键/字段镜像契约钉在 `tests/mcp/test_telemetry_field_rules.py`）；字符串级 `redact_uuids` 全事件 UUID 掩码（mlflow run_id 不中枪）。只作用于上报副本，工具真实返回零影响；例外清单单源见 PRIVACY.md
- **数据去向**：默认项目官方遥测网关（TLS）；`MFLOWY_TELEMETRY_ENDPOINT` 指向自建网关（自建部署见 `docker/telemetry/`）；设备密钥 `~/.mflowy/api.key`（0600）本机生成，服务端只存 hash
- **边界**：端点不可达完全透明（不阻塞、静默重试/丢弃）；仅覆盖 MCP 入口（`mflowy-mcp`），CLI（`mflowy`）与直接 import 不采集
- 实现：`mcp/telemetry.py`（唯一接线点 `wire_agentcat`，禁止 per-tool 打点）；隐私契约（采集面与脱敏例外清单）见 [PRIVACY.md](../../PRIVACY.md)

## 相关文档

- DAG 内核与插件 SDK：[mflowy-driver README](../driver/README.md)
- 能力目录与边函数：[mflowy-builtin-plugins README](../builtin_plugins/README.md)
- 远程执行（JobProvider 协议与自定义执行环境接入）：[mflowy-model README](../builtin_plugins/model/README.md)
