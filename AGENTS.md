# AGENTS.md

AI 编码 agent 在本仓库的唯一指引。代码是真相源——本文件只写代码里查不到的东西：约定、理由、陷阱、流程。存疑先探索再假设；动态文件树用 `tree --gitignore` 现查。

## Project

MFlowy：YAML 配置驱动的模块化 ML 工作流引擎，全部能力以 **MCP tools**（pyfunc）暴露——MCP server（stdio）、JSON runner（K8s Job 容器）、直接 import 三入口。

**Language**: Python 3.12 | 代码注释与文档：中文

## 开发命令

```bash
uv sync --all-extras --all-groups   # 首次/依赖变更后
make test                           # 全量测试
make lint && make fmt               # ruff
uv run cmd list_modules                          # base 能力（无数据栈）
uv run --extra modeling cmd modeling '{"modeling_steps_yaml": "...", "name": "...", "desc": "..."}'
uv run mcpSrv                                     # MCP server（stdio）
```

用户态分发（PyPI/uvx/wheel/Docker）见 [README](README.md) 与 `docker/`；MCP 客户端 stdio 配置见 [`.mcp.json.example`](.mcp.json.example)。

**uvx 陷阱**（uvx 不读 pyproject 的 `[tool.uv.sources]`）：

- torch CPU 索引（`--index https://download.pytorch.org/whl/cpu` + `unsafe-best-match`）为 [modeling] 必需——缺省时 torch 解析为 CUDA 全家桶（2–3GB）
- extras 内联进 `--from "mflowy[modeling]"`——uvx 的 `--extra` 需新版 uv
- 入口名 `mcpSrv` 刻意避开 mcp SDK 同名 CLI（`mcp.cli:app`）——uvx 解析 `mcp` 命令可能命中 SDK 侧启动失败

## 分层

uv workspace 五 distribution（PEP 420 namespace，`mflowy` 聚合包 + 四子包，版本锁步，`tests/test_workspace.py` 断言）：

| 层          | 位置                             | 职责                                  |
| ----------- | -------------------------------- | ------------------------------------- |
| MCP Tools   | `packages/mcp/src/mflowy/mcp/`   | 全部工具定义、stdio server、runner、JobProvider 执行策略 |
| Driver      | `packages/driver/src/mflowy/driver/` | YAML 解析、DAG 调度、@handler 插件 SDK、entry points 发现；默认尾链在 `builtin_middleware.py` |
| Plugins     | `packages/builtin_plugins/src/mflowy/builtin_plugins/` | 内置能力 + 注入器（`middlewares/`：Get*/inject*/log_*）+ 第三方插件的活参考实现 |
| Utils       | `packages/utils/src/mflowy/utils/`   | mlflow/optuna/jinja/日志/捕获工具层    |

依赖方向单向：utils ← driver ← builtin_plugins ← mcp；**driver/utils/mcp 不得 import builtin_plugins**（测试断言）。

工具三分：建模类 4 种（modeling / explanation / predict / inverse_optimization）经 JobProvider 委派；分析类 3 种（data_profile / eda / infer_task_type_by_statistic）与 info/mlflow 组始终本地执行。

- **零手工注册**：插件模块入口函数加 `@handler(中间件...)` 即完成标记；身份由构建期 `hatch_metadata.py` 扫描生成的 entry point（`step.module`）声明，运行期 `driver/discover.py` 按元数据解析（`_` 前缀与 `_EXCLUDED` 词干除外）
- **通道边界**：`print`（stdout）= 业务数据，per-task 捕获进 `NodeResult.output`；`logger`（stderr）= 过程诊断。同一信息只走一条通道
- 模块内相对导入，跨模块绝对导入；KISS
- 测试随包：`packages/<pkg>/tests/**` 镜像包内 `mflowy/<pkg>/**` 路径；`tests/` 根放 integration + workspace 完整性断言；根 `conftest.py` 全仓共享 fixtures

动这些前先读：driver / YAML 语义 → [docs/DRIVER.md](docs/DRIVER.md)；远程 provider 接入 → [docs/REMOTE_MODELING.md](docs/REMOTE_MODELING.md)；遥测采集与同意门 → [docs/TELEMETRY.md](docs/TELEMETRY.md)。

## 添加能力

**新插件模块**：`packages/builtin_plugins/src/mflowy/builtin_plugins/<step>/` 建模块，入口函数加 `@handler(中间件...)`。判据：镜像路径有对应测试，`uv sync` 后 MCP `list_modules` 可见（entry points 在安装时生成，新模块必须重跑 `uv sync`）。新能力族额外在 `hatch_metadata.py` 的 `_STEP_OF_DIR` 补一行目录→step 映射——漏配时构建直接报错，不会静默丢能力。

**第三方插件包**（对外的活参考 = `packages/builtin_plugins/`）：以 `mflowy` 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]`（name 格式 `step.module`），随 `uv --with` / 镜像 `MFLOWY_EXTRA_MODULES` 安装即生效。

**新 JobProvider 委派的 compute 工具（5 站点，缺一契约即断）**：

1. `packages/mcp/src/mflowy/mcp/job_provider/protocol.py` — 方法签名（`headers` 形参 = 请求 `_meta` 直传，runner 场景恒 None）
2. `packages/mcp/src/mflowy/mcp/job_provider/local.py` — 编排实现
3. 远程 provider 实现（module:Class 自定义包）— build command + `_dispatch`
4. `packages/mcp/src/mflowy/mcp/tools.py` — 委派函数 + docstring（schema 由此编译）
5. `packages/mcp/src/mflowy/mcp/server.py` — 注册工具

判据：`tools/list` 出现新工具且 `tests/mcp` 契约测试通过。本地分析类工具不经 JobProvider，只需 4/5 两站。

## 文档模型

仓库只保留两类 markdown：根级治理文件（README / CONTRIBUTING / SECURITY / PRIVACY / CODE_OF_CONDUCT / CHANGELOG / 本文件）与 `docs/` 专题文件——

| 触发场景 | 去处 |
| -------- | ---- |
| 用户方法论 / 写案例叙事 | [docs/research-flow.md](docs/research-flow.md)（操作手册；代码讨论的依据是代码与专题文档） |
| 规划新功能 / 查路线图状态 | [docs/roadmap.md](docs/roadmap.md) |
| 写远程 provider / K8s Job 接入 | [docs/REMOTE_MODELING.md](docs/REMOTE_MODELING.md) |
| 动遥测 / 采集范围 / 同意门 | [docs/TELEMETRY.md](docs/TELEMETRY.md) |
| 动 driver / 或 YAML 语义 | [docs/DRIVER.md](docs/DRIVER.md) |

**防漂移**（文档只写终态）：

1. 代码已表达的（签名 / 清单 / 文件树 / 行号）只链接不复述；禁行号锚点与静态文件树
2. 只写追加式文档：新案例 → `examples/YYYY-MM-DD-*.md` 快照；新版本 → `CHANGELOG.md`
3. MCP schema 即 API 文档：工具行为变化改 docstring（经 `tools/list` 自动下发）
4. 漂移可接受，release 边界批量回填
5. 新建文档先过漂移测试——"这文件里哪句话会因下次代码 PR 失效？"有答案就不建

## Issue Tracker（gh CLI）

repo 从 `git remote -v` 自动推断。issue/PR 共号段：`#42` 先 `gh pr view 42` 再 `gh issue view 42`；外部 PR 不走 triage 流程。

**Triage 五标签**：`needs-triage`（待评估）/ `needs-info`（等补充）/ `ready-for-agent`（交 AFK agent）/ `ready-for-human`（需人实现）/ `wontfix`。

**Wayfinding**（`/wayfinder` 用）：map = 单个 `wayfinder:map` label issue；child = sub-issue（降级：map body task list + `Part of #<map>` 行）；blocking = 原生 issue dependencies（降级：child 顶部 `Blocked by: #<n>` 行）；claim = `--add-assignee @me`；resolve = comment + close + 回写 map 的 Decisions-so-far。

## 词表

**唯一词表 = 运行期插件目录**（`driver/discover.py` 读取 `mflowy.builtin_plugins` / `mflowy.plugins` 两组 entry points，name 格式 `step.module`）。词表可在构建/编译期校验：YAML 中的 step 以 `Builder.build()` 对目录全集校验，未知值 fail-loud。issue 标题 / 重构提案 / 测试名一律用 step 术语；目录里没有的概念 = 语言超纲信号——回 `hatch_metadata.py` 加能力族映射或换词。第三方包以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]` 即成插件（`driver/handler.py` 装饰器、`builtin_middleware.py` 与 `builtin_plugins/middlewares/` 的注入器即插件 SDK，破坏性变更受 CHANGELOG 语义化版本约束）。
