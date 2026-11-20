# mflowy-app — 应用层：CLI 入口 · 工具目录 · 平台登录 · 遥测

MFlowy 全部能力以工具（pyfunc）暴露，经 entry points 组 `mflowy.tools` 发现——域工具住各自域包（mflowy-app 硬依赖三域包，#243 单一安装形态：装 mflowy-app 即全词表），平台工具住本包，三种入口共享同一套实现。

| 入口 | 命令 | 场景 |
|------|------|------|
| **CLI 通道** | `mflowy <tool> [--flags]`（或 `--kwargs-json '<JSON>'` 逃生舱） | 命令行、agent（subprocess 并行调用 = 并行进程无 GIL 争用）；`DAGResult.status != finished` 以**非零退出码**外传失败，不吞 |
| **容器入口** | `mflowy-job wrap --tool <t> --kwargs-json <j> [--inherit-from <job>]` | 平台 Job 容器（#158：继承解包 → 执行 → 无条件 zip 归档，工具 rc 透传） |
| 直接 import | `mflowy.{app,data,model,bo}.tools` pyfunc | 宿主程序内嵌调用 |

## CLI（#243）

旗标从 pyfunc pydantic 签名**编译**（单源：签名 = 约束 = 帮助 = 校验）：

- `mflowy --help` = 按所属域分组的工具清单 + 一行「何时使用」介；`mflowy <tool> --help` = usage / 何时（docstring 全文）/ options 三段
- 标量 → 类型化旗标（`--max-steps 15`）；bool → `--flag/--no-flag`；`list`/`dict`/嵌套模型 → JSON 字符串旗标（`--prior '{"target": "y"}'`）；复杂模型的字段面在 help 展开（CleanPrior/CVCard 一跳可发现）
- `--kwargs-json '<JSON 对象>'` 全量逃生舱（jobwrap 先例）；与具名旗标同键 = 参数来源冲突拒绝
- 旗标层零校验，pydantic 运行时校验 + coerce（动态 `create_model`，未知键 fail-loud）
- 遥测元旗标 `--telemetry-session-id` / `--telemetry-intent`：CLI 层剥离、不进工具签名，调用方主动注入

**工件锚定（task_dir）**（#95）：工具以 `@anchor_task_dir("<参数名>")` 声明锚参数（纯元数据），从业务路径参数推父目录为任务目录（实验工件根 = `task_dir/.mlruns`）；`MFLOWY_TASK_DIR` env **钉住**（推导不覆写）→ 未设时推导 → 缺省 `cwd`；相对入参路径按执行 cwd 解析。

> **平台工具静态声明**：平台工具（词表/mlflow 观测/工件/指纹/登录/平台 API）由 mflowy-app
> pyproject **同组静态声明**（与域包同构，CLI 分发面同一真源消费，镜像契约钉在
> `tests/app/test_platform_tools_registry.py`）。remote 系列平台工具
> （`ensure_platform_experiment` 等）的引擎住 model 包 platform_client，**函数体内懒加载**。

## 工具分组（执行位置）

| 类别 | 工具 | 执行位置 |
|------|------|---------|
| info | `list_nodes` / `get_node_info` / `file_hash` | 始终本地（base 依赖够用） |
| 分析与问卷 | `data_profile` / `eda` / `propose_questions` | 始终本地（轻量、无跨环境状态） |
| POMDP 会话 | `data_*` 六操作 + `modeling_*` 七操作（含 `report_findings`） | 始终本地（会话态在本地盘上工件） |
| 报告素材 | `draw_taylor_diagram` / `modeling_report_skeleton` | 始终本地 |
| YAML 预检 | `validate_modeling_steps` | 始终本地（结构化 `{valid: false, error}` 错误契约） |
| 建模 compute | `modeling` / `hyperparameters_search` / `explanation` / `predict`（受理面 = JobOutcome：finished 内嵌 DAGResult / submitted 提交凭据） | **JobProvider 委派**（local / remote） |
| 执行模式 | `switch_job_provider`（model 包——写 `~/.mflowy/settings.json` 的 `job_provider` 键，解析序 env `MFLOWY_JOB_PROVIDER` 钉住 > settings > 缺省 local） | 本地 |
| mlflow 观测 | `get_experiment` / `get_run`（可选 `tracking_uri` 单参——远程产物下载库查询，工件路径自动改写） | 始终本地（工件后端为本地文件树） |
| 平台 | `login`（OAuth 2.1 浏览器授权）/ `sessions_gc`（TTL 回收） | 本地 |
| 平台·remote | `ensure_platform_experiment`（实验就绪：查建 + 文件增量同步 + 可继承面） | 平台 API（经 model 包 platform_client，懒加载） |
| 平台·remote | `get_platform_job_logs`（stateless 轮询：status/terminal/logs 合并返回） | 平台 API（经 model 包 platform_client，懒加载） |
| 平台·remote | `download_platform_job_artifacts`（产物拉取：zip 解压 `target_dir/{job_id}/` 原子替换；mlflow.db 供观察面 `tracking_uri` 单参查询） | 平台 API + 本地落盘 |
| BO 推荐 | `bo_load_or_start` / `bo_ask` / `bo_prune` / `bo_diversity_check` / `bo_tell` / `bo_export_dataset` | 始终本地（study sqlite db 盘上） |

建模类经 JobProvider 委派（内置 `local` / `remote`、自定义 `module:Class` 仅经 env 预配置）——协议与自定义执行环境接入见 [mflowy-model README](../builtin_plugins/model/README.md)「JobProvider」节（预测模型训练子系统）。

**错误契约**：结构化失败面（`validate_modeling_steps` 的 `{valid: false, error}`）之外，工具的 fail-loud 报错（红线 1 的候选清单式文案——路径/run 不存在、参数冲突等）在 CLI 通道全量打印（stderr）+ 非零退出码；参数/旗标形态错误 exit 2、工具执行错误 exit 1、成功 exit 0。

## 平台登录（login）

远程工具（经 JobProvider 回调平台 API）需要平台身份：`login` 一次调用完成 OAuth 2.1 授权码流——PKCE S256 + 本机浏览器授权 + loopback 回调（127.0.0.1 随机端口）+ 凭据原子落盘；授权链接恒打印终端（stderr——浏览器未自动拉起时可复制手动打开）。契约对端为平台 OAuth 授权服务器（快照见平台仓 `docs/OAUTH_CLI.md`）。

- **幂等**：已有同环境凭据直接返回 `already_logged_in` 摘要（不弹浏览器）；`force=true` 强制重授权
- **scope 缺省**：登录时拉服务端元数据 `scopes_supported` 全目录整体同意——客户端不拷目录，服务端目录变更零漂移；显式传 `scopes` 可收窄
- **环境**：`base_url` 参数 > `MFLOWY_BASE_URL` env > `http://localhost:30080`（本地 gateway NodePort；生产/自建网关经 env 指向）
- **凭据**：`~/.mflowy/credentials.json`（0600）单文件整替——refresh 为轮换制，旧值复用会触发平台复用检测（账号全端下线），故刷新/落盘临界区持文件锁（同机多 agent host 并发安全）
- **消费接线**：`mflowy.utils.auth.ensure_access_token()` / `LoginRequiredError`——远程 JobProvider 以此附 Bearer（401 单次刷新重试），不重造轮子
- **边界**：首版需本机浏览器（headless/SSH 场景经终端打印的授权链接手动打开）；JWT 本地仅解码展示不验签（资源侧统一验）
- **登录即同意**：`mflowy login` 成功无条件写 `telemetry=true`（见下节）

实现：`mflowy/utils/auth.py`（纯 stdlib）；行为契约见 `packages/utils/tests/test_auth.py`。

## 遥测与同意门

CLI 工具调用诊断采集：工具名、完整输入输出、调用意图。手写 OTLP JSON 发送器直出配置端点（默认官方网关；agentcat 与 MCP SDK 已随 #243 退役）——每调用后台线程 fire-and-forget、短超时、失败静默（绝不影响工具调用与退出码）。

- **同意模型（TTY 分流）**：`~/.mflowy/settings.json` 的 `telemetry` 键缺省时——TTY 首次 y/n 询问并落盘（此后不再问）；非 TTY（agent 场景）默认开 + stderr 单行关闭指引；`false` = 关；`mflowy login` 成功**无条件写 `telemetry=true`**（登录即同意）；`MFLOWY_TELEMETRY=on|off` 显式覆盖（容器 / CI 等无交互场景的唯一入口），无条件优先于 settings.json
- **session 关联**：session_id / intent 仅来自调用方经 CLI 元旗标注入；会话寻址锚 `observation_json`（状态文件本机绝对路径）不进遥测
- **遥测副本脱敏**（#189，fail-closed）：事件级按字段规则表 `TELEMETRY_FIELD_RULES` 剔除标识字段（表键/字段镜像契约钉在 `tests/app/test_telemetry_field_rules.py`）；字符串级全事件 UUID 掩码（mlflow run_id 不中枪）。只作用于上报副本，工具真实返回零影响；例外清单单源见 PRIVACY.md
- **数据去向**：默认项目官方遥测网关（TLS）；`MFLOWY_TELEMETRY_ENDPOINT` 指向自建网关（自建部署见 `docker/telemetry/`）；设备身份 = py-machineid + 设备密钥 `~/.mflowy/api.key`（0600，本机生成、服务端只存 hash，401/403 自动重签）
- **边界**：端点不可达完全透明（不阻塞、静默丢弃）；覆盖 CLI 入口（`mflowy`），直接 import 与 `mflowy-job` 容器入口不采集
- 实现：`app/telemetry.py`（每调用入口 `report_call`，禁止 per-tool 打点）；隐私契约（采集面与脱敏例外清单）见 [PRIVACY.md](../../PRIVACY.md)

## 相关文档

- DAG 内核与插件 SDK：[mflowy-driver README](../driver/README.md)
- 数据域能力目录：[mflowy-data README](../builtin_plugins/data/README.md) · 建模域（含 JobProvider 协议与自定义执行环境接入）：[mflowy-model README](../builtin_plugins/model/README.md) · BO 域：[mflowy-bo README](../builtin_plugins/bo/README.md)
