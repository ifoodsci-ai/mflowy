# AGENTS.md

AI 编码 agent 在本仓库的唯一指引。代码是真相源——本文件只写代码里查不到的东西：约定、理由、陷阱、红线。存疑先探索再假设；动态文件树用 `tree --gitignore` 现查。

**同步规则**：本文件随行为变更同 PR 更新（工具签名、依赖方向、注册站点、迁移表状态）。教学文档漂移 = 缺陷——用户文档可以等 release 回填，本文与 `.agents/skills/` 的漂移按 P0 处理。

## Project

MFlowy：YAML 配置驱动的模块化 ML 工作流引擎，全部能力以 **MCP tools**（pyfunc）暴露——MCP server（stdio）、JSON runner（K8s Job 容器）、直接 import 三入口。

**Language**: Python 3.12 | 代码注释与文档：中文

## 开发命令

```bash
uv sync --all-packages --all-extras --all-groups   # 首次/依赖变更后；新增插件模块后必跑（entry points 安装期生成）
make test                           # 全量测试
make lint && make fmt               # ruff
uv run mflowy list-nodes            # 能力词表
uv run mflowy-mcp                   # MCP server（stdio）
```

用户态分发（PyPI/uvx/wheel/Docker）见 [README](README.md) 与 `docker/`；MCP 客户端 stdio 配置见 [`.mcp.json.example`](.mcp.json.example)。

**uvx 陷阱**（uvx 不读 pyproject 的 `[tool.uv.sources]`）：

- torch CPU 索引（`--index https://download.pytorch.org/whl/cpu` + `unsafe-best-match`）为 [deep] 必需（仅 MLP；`[model]` CPU 栈干净安装）——缺省时 torch 解析为 CUDA 全家桶（2–3GB）
- extras 内联进 `--from "mflowy[model]"`——uvx 的 `--extra` 需新版 uv
- 入口名 `mflowy-mcp` 刻意避开 mcp SDK 同名 CLI（`mcp.cli:app`）

## 分层

uv workspace 六 distribution（PEP 420 namespace，版本锁步，`tests/test_workspace.py` 断言）。extras = 域选择器：聚合包 `[data]`/`[model]`/`[mcp]`/`[deep]`（torch 栈隔离在 model 包 `[deep]`，唯一可选 extra）。

| 层     | 位置                              | 职责                                                       |
| ------ | --------------------------------- | ---------------------------------------------------------- |
| MCP    | `packages/mcp/mflowy/mcp/`        | 工具注册面、stdio server、runner、遥测                     |
| Data   | `packages/builtin_plugins/data/mflowy/data/`      | 清洗 MDP 动作域（load/statistic/detect/clean/plot + DomainCard） |
| Model  | `packages/builtin_plugins/model/mflowy/model/`    | 建模 MDP 动作域（x_y/cross_validation/x_transformer/model + plot）+ JobProvider + diagnose/report 观察面 |
| Driver | `packages/driver/mflowy/driver/`  | YAML→DAG 编译调度、@handler/@edge 插件 SDK、MDP 会话基座、规则引擎 SDK |
| Utils  | `packages/utils/mflowy/utils/`    | mlflow seam、脚本安全扫描、capture/path/logging/jinja      |

依赖方向保持单向：utils ← driver ← data ← model ← mcp（内核零域感知、两 MDP 域隔离——`tests/test_workspace.py` 断言，改动 import 保持全绿）。模块内相对导入、跨模块绝对导入。

工具六组共 31 个（真源 = `tools/list`）：平台 8（info 3 + mlflow 观测 4 + 会话回收 1，本地）；本地分析 4（`data_profile`/`eda`/`propose_questions`/`taylor_diagram_report`）；YAML 预检 1（`validate_modeling_steps`）；compute 5（`hyperparameters_search`/`modeling`/`explanation`/`predict`/`inverse_optimization`，经 JobProvider 委派，协议在 model 包）；MDP 会话 13（`clean_*`×6 + `modeling_*`×7——含 `modeling_report_findings`，引擎在两域 session 模块，动作面 = modeling 八动作面）。

插件身份零手工注册：`@handler(namespace=...)` 标记（裸命名参数=参数边，Annotated=用户可配），构建期 `hatch_metadata.py` AST 扫描生成 entry point，运行期 `driver/discover.py` 按元数据解析（`_` 前缀与 `_EXCLUDED` 词干除外）。

## 质量红线（每个变更遵守）

1. **fail-loud**：未知键、撞名、缺依赖一律 raise 并带候选清单。凡是"静默覆盖 / 静默丢弃 / 白名单过滤后继续"的写法都是缺陷。参数来源冲突（YAML params 撞参数边名、用户 pin 撞引擎管理轴）显式报错——宁可拒绝一次合法输入，绝不静默改写语义。
2. **单源**：一个事实一处定义。签名=约束=文档=校验（同一台 pydantic 机器）；词表=运行期目录；默认值=函数签名。第二份拷贝出现即收敛（提共享件），任其分叉即债。
3. **原子提交**：manifest/版本工件落盘走 tmp+rename；保持"先写工件、后写 manifest"的 commit 顺序；回滚把内存态与盘上工件一起回退——只回退一半 = 不一致交付。
4. **通道纪律**：`print`（stdout）= 业务数据，只出现在捕获窗口内（`capture_prints`/redirect——MCP stdio 下裸 print 毁掉 JSON-RPC 通道）；`logger`（stderr）= 过程诊断。同一信息只走一条通道。
5. **缺依赖=优雅降级**：extra 缺失时词表/会话对相关节点降级标注（`bindings=None`/跳过+warning），其余能力照常；词表 miss 与 import 失败按类型二分（`CatalogMiss` vs 裸 `ModuleNotFoundError`）。
6. **依赖声明完整**：import 了什么就声明什么——靠传递依赖侥幸工作的 import 是缺陷（当前存量：scipy）。
7. **测试钉契约**：行为变更带契约测试；架构不变量（依赖方向/版本锁步/注册表镜像）住 `tests/`；测试替身只打 adapter 缝位（`_DEFAULT_CATALOG`/`Tracking` seam）；测试随包镜像 `packages/<pkg>/tests/** ↔ mflowy/<pkg>/**`，跨域接缝用例住消费侧包；接缝测试真跑产物，以 mock 为主的地方补真跑。
8. **文档同频**：工具行为改 docstring（`tools/list` 自动下发即 API 文档）；本文件与 skills 随行为变更同 PR 更新（见同步规则）。

## 目标态与迁移规则（破坏性授权）

下表为目标态，优先级高于现状兼容：触碰违规区域的变更顺手收敛（可 breaking，CHANGELOG 记 BREAKING，验收测试先行），也接受专项 PR。完成一条即更新本表行与 CHANGELOG。

| #   | 目标态                                                                 | 存量违规点                                        | 验收                                                   |
| --- | ---------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------ |
| M3  | `SessionStore.create` 只建目录；两域 `_is_terminal` 委托基座 `is_terminal_state` | driver 侧已落地（`write_manifest` tmp+rename / 共享单源 / gc root 守卫，见 CHANGELOG）；余：两域绕过 create、`_is_terminal` 逐字重复 | 两域 `_is_terminal` 逐字重复删除                      |
| M4  | 会话壳收敛：store/终态/decision_log 形状/plan-YAML emitter 住 driver，域只填转移语义 | 两域 ~1/5 骨架逐字重复、plan YAML 手拼分叉        | 两域 session 各缩 ~150 行；`MDPSessionHandle` 结构等价有测试钉住 |
| M5  | 版本树 append-only（`current_v` 指针）；loader 工件带版本号，rollback 同步盘上 | clean 截断覆写弃分支；model loader 覆写不同步      | rollback-after-rerun 一致性用例                       |
| M6  | `registry.families()` 对缺 extra 条目降级跳过                          | 无守卫，CPU-only 开不了建模会话                    | 无 torch 环境 CI job 跑通 RF 会话                     |
| M7  | 会话引擎路径全捕获（`_detect_all` 包 capture）                         | detect 算子裸 print 污染 stdio                    | MCP stdio e2e 断言纯 JSON-RPC 输出                    |
| M9  | 遥测端点 env 可覆盖 + agentcat monkeypatch 带 feature-detect           | 端点硬编码；私有内部 monkeypatch 静默失效          | `MFLOWY_TELEMETRY_ENDPOINT` 指向 `docker/telemetry` 可用 |
| M10 | 重栈产物锚定 task_dir/`~/.mflowy`；仓库根与包根零运行产物              | lightning_logs/mlruns/catboost_info 落包根        | `make test` 后 `git status` 干净                       |
| M11 | 剧本层完整入库：skills 全 git 追踪、引用文档入仓、工具名 ⊆ `tools/list` 有测试锁定 | research-review 未追踪；引用被 ignore 的文档断链  | `test_workspace.py` 增 SKILL 工具名断言               |
| M12 | breaking 小步走 0.x minor，0.2.0 尽快切 released                       | unreleased 堆 81 commit 三批 BREAKING             | CHANGELOG 0.2.0 带 release 日期                        |

小项随触碰收敛（不单列）：finish 与过程奖励同尺（卡权重）、propose 只出签名可受理的候选、`utils` mlflow 转 extra、域专用查询迁出 utils、runner 分发表与 server 注册清单同源。

## 添加能力

**新插件节点**：按域建模块（`packages/<data|model>/mflowy/<pkg>/<能力族>/`），入口加 `@handler(namespace=...)`（归组）或 `@handler()`（非归组）；纯变换边加 `@edge`（edges ns）。判据：镜像路径有测试 + `uv sync` 后 `list-nodes` 可见。

**新 compute 工具**（JobProvider 委派，7 站点缺一即断）：

1. `packages/builtin_plugins/model/mflowy/model/job_provider/protocol.py` — 方法签名（`headers` 形参 = 请求 `_meta` 直传，runner 场景恒 None）
2. `packages/builtin_plugins/model/mflowy/model/job_provider/local.py` — 本地编排实现
3. 远程 provider 实现（module:Class 自定义包）— build command + `_dispatch`
4. `packages/builtin_plugins/model/mflowy/model/tools.py` — 工具包装 + docstring（schema 由此编译）
5. `packages/builtin_plugins/model/pyproject.toml` `[tool.mflowy.tools]` — 注册条目
6. `packages/builtin_plugins/model/mflowy/model/templates.py` + `templates/*.yaml.j2` — 模板
7. `packages/builtin_plugins/model/tests/` — headers 契约 + provider 用例；工具清单从 `JobProvider` 协议自省生成（单源）

判据：`tools/list` 出现新工具且镜像/契约测试通过。本地分析类工具走 4/5 两站（server 经 entry points 自动注册）。

**第三方插件包**（活参考 = `packages/builtin_plugins/data/` + `packages/builtin_plugins/model/`）：以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]`（`namespace.node` / 裸 `node`），随 `uv --with` / 镜像 `MFLOWY_EXTRA_MODULES` 安装即生效。

## 文档模型

三类 markdown：根级治理文件、组件文档（`packages/<pkg>/README.md`，随包分发）、`docs/` 跨组件专题。路由（触发词打头）：

| 触发场景                   | 去处                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------- |
| 查术语 / 选词              | [CONTEXT.md](CONTEXT.md)（引擎元词汇 + 禁用别名）                                      |
| 动 driver / YAML / 写插件  | [packages/driver/README.md](packages/driver/README.md)                                 |
| 查能力目录 / 边函数 / 遥测 | [packages/builtin_plugins/data/README.md](packages/builtin_plugins/data/README.md) + [packages/builtin_plugins/model/README.md](packages/builtin_plugins/model/README.md) |
| 写远程 provider / 遥测同意门 | [packages/mcp/README.md](packages/mcp/README.md)                                     |
| 用户方法论 / 案例叙事      | [docs/research-flow.md](docs/research-flow.md)                                         |
| 规划新功能 / 查路线图      | [docs/roadmap.md](docs/roadmap.md)（已实现清单以 `tools/list` 为真源，roadmap 只留规划） |

**防漂移**：代码已表达的只链接不复述（文件树/签名/行号锚点留给代码）；只写追加式文档（新案例 → `examples/` 快照，新版本 → `CHANGELOG.md`）；"漂移可接受"仅限用户文档，release 边界回填——本文件与 skills 除外（见同步规则）。新建文档先过漂移测试："这文件里哪句话会因下次代码 PR 失效？"有答案就不建。

## Issue Tracker（gh CLI）

repo 从 `git remote -v` 推断。issue/PR 共号段：`#42` 先 `gh pr view 42` 再 `gh issue view 42`；外部 PR 不走 triage。

**Triage 五标签**：`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`。

**Wayfinding**（`/wayfinder` 用）：map = `wayfinder:map` label issue；child = sub-issue（降级：map body task list + `Part of #<map>`）；blocking = 原生 dependencies（降级：`Blocked by: #<n>` 行）；claim = `--add-assignee @me`；resolve = comment + close + 回写 map 的 Decisions-so-far。

## 词表

两层：**引擎元词汇**（节点/边/组/Context 的定义与禁用别名）见 [CONTEXT.md](CONTEXT.md)；**节点名唯一词表 = 运行期插件目录**（`driver/discover.py` 读两组 entry points；边注册于 `edges` ns 但不进 `list_nodes`）。**编译即校验**：Builder 增量编译（平铺有序列表 + 节点表）对参数边全量 fail-loud——拓扑序/词表 miss/撞名/类型门。issue 标题 / 提案 / 测试名一律用 node/namespace 术语；目录里没有的概念 = 语言超纲信号。插件 SDK = `driver/handler.py` + `edge.py` + `builtin_middleware.py` 尾链，破坏性变更受 CHANGELOG 语义化版本约束。
