# AGENTS.md

AI 编码 agent 在本仓库的唯一指引。代码是真相源——本文件只写代码里查不到的东西：约定、理由、陷阱、红线。存疑先探索再假设；动态文件树用 `tree --gitignore` 现查。

**同步规则**：本文件随行为变更同 PR 更新（工具签名、依赖方向、注册站点）。教学文档漂移 = 缺陷——用户文档可以等 release 回填，本文与 `.agents/skills/` 的漂移按 P0 处理。

## Project

MFlowy：YAML 配置驱动的模块化 ML 工作流引擎，全部能力以 **MCP tools**（pyfunc）暴露——MCP server（stdio）、JSON runner（K8s Job 容器）、直接 import 三入口。业务系统 = PILOT 三子系统闭环（[docs/pilot-loop.md](docs/pilot-loop.md)）：Clean / Model 两域已落地，BO 推荐域设计进行中（#171）。

**Language**: Python 3.12 | 代码注释与文档：中文

## 开发命令

```bash
uv sync --all-packages --all-extras --all-groups   # 首次/依赖变更后；新增插件模块/注册条目后必跑（entry points 随安装元数据生效）
make test                           # 全量测试
make typecheck                      # pyright basic 类型门（packages+tests；压制纪律见 #91）
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
| Data   | `packages/builtin_plugins/data/mflowy/data/`      | 清洗 POMDP 动作域（load/statistic/detect/clean/plot + DomainCard 先验分级 + 缺陷封闭枚举/Finding 契约 + 路由提示表） |
| Model  | `packages/builtin_plugins/model/mflowy/model/`    | 建模 POMDP 动作域（x_y/cv/x_transformer/model + plot——cv 族 = split 参数构建/load 引用加载）+ JobProvider + diagnose/report 观察面 |
| Driver | `packages/driver/mflowy/driver/`  | YAML→DAG 编译调度、@node/@edge 插件 SDK、POMDP 会话基座（`pomdp` 契约 + `session` 壳——六操作 start/observe/apply/rollback/finish/abandon） |
| Utils  | `packages/utils/mflowy/utils/`    | mlflow seam、OAuth 2.1 客户端（login/token 库）、脚本安全扫描、capture/path/logging/jinja |

依赖方向保持单向：utils ← driver ← data ← model ← mcp（内核零域感知、两 POMDP 域隔离——`tests/test_workspace.py` 断言，改动 import 保持全绿）。模块内相对导入、跨模块绝对导入。

工具六组共 32 个（真源 = `tools/list`）：平台 8（info 3 + mlflow 观测 2（`get_experiment`/`get_run`）+ 会话回收 1 + 登录 1，本地；实验就绪 1（`ensure_platform_experiment`，平台 API 面——引擎住 model 包 platform_client 懒加载，`mflowy[mcp]` 聚合含 model））；本地分析 5（`data_profile`/`eda`/`propose_questions`/`draw_taylor_diagram`/`modeling_report_skeleton`）；YAML 预检 1（`validate_modeling_steps`）；compute 5（`hyperparameters_search`/`modeling`/`explanation`/`predict`/`inverse_optimization`，经 JobProvider 委派，协议在 model 包）；POMDP 会话 13（`clean_*`×6 + `modeling_*`×7——含 `modeling_report_findings`，引擎在两域 session 模块，动作面 = modeling 八动作面）。

插件身份静态声明：`@node(namespace=...)` 标记（裸命名参数=参数边，Annotated=用户可配）+ 域包 pyproject `[project.entry-points."mflowy.builtin_plugins"]` 手写条目（0.2.0 起构建期 AST 扫描钩子退役，与第三方包声明方式同构），运行期 `driver/discover.py` 按注册表（装饰器写入 `<node_key, Node>`，#126 零属性挂载）解析——发现表键 = 注册表键，声明与注册不一致（未装饰/错 namespace/别名键）加载期 fail-loud（#140 身份桥退役）；忘声明由 CR 交叉检查兜底（见「添加能力」）。

## 质量红线（每个变更遵守）

1. **fail-loud**：未知键、撞名、缺依赖一律 raise 并带候选清单。凡是"静默覆盖 / 静默丢弃 / 白名单过滤后继续"的写法都是缺陷。参数来源冲突（YAML params 撞参数边名、用户 pin 撞引擎管理轴）显式报错——宁可拒绝一次合法输入，绝不静默改写语义。
2. **单源**：一个事实一处定义。签名=约束=文档=校验（同一台 pydantic 机器）；fn 反射（typing/inspect）只居 `driver/reflect.py`，装饰期冻结进 Node/Edge 契约面、消费方读冻结核（#138，工具包外存量台账由 `tests/test_reflection_core.py` 锁定）；词表=运行期目录；默认值=函数签名。第二份拷贝出现即收敛（提共享件），任其分叉即债。
3. **原子提交**：manifest/版本工件落盘走 tmp+rename；保持"先写工件、后写 manifest"的 commit 顺序；回滚把内存态与盘上工件一起回退——只回退一半 = 不一致交付。
4. **通道纪律**：`print`（stdout）= 业务数据，只出现在捕获窗口内（`capture_prints`/redirect——MCP stdio 下裸 print 毁掉 JSON-RPC 通道）；`logger`（stderr）= 过程诊断。同一信息只走一条通道。
5. **缺依赖=优雅降级**：extra 缺失时词表/会话对相关节点降级标注（`bindings=None`/跳过+warning），其余能力照常；词表 miss 与 import 失败按类型二分（`CatalogMiss` vs 裸 `ModuleNotFoundError`）。
6. **依赖声明完整**：import 了什么就声明什么——靠传递依赖侥幸工作的 import 是缺陷（当前存量：scipy）。
7. **测试钉契约**：行为变更带契约测试；架构不变量（依赖方向/版本锁步/注册表镜像）住 `tests/`；测试替身只打 adapter 缝位（`_DEFAULT_CATALOG`/`Tracking` seam）；测试随包镜像 `packages/<pkg>/tests/** ↔ mflowy/<pkg>/**`，跨域接缝用例住消费侧包；接缝测试真跑产物，以 mock 为主的地方补真跑。
8. **文档同频**：工具行为改 docstring（`tools/list` 自动下发即 API 文档）；本文件与 skills 随行为变更同 PR 更新（见同步规则）。
9. **mixin 组合方契约**：mixin 引用组合方才满足的属性/方法，一律直取 + 带码 ignore 注明「组合方满足——定义期不可见」（#122 裁定：是否满足取决于组合方）；TYPE_CHECKING 借基/伪继承伪造类型面即债。属性在缺是值级可选性（`getattr(..., None)` 语义，如 `evals_result_` 随 eval_set 在缺）；第三方无 py.typed 的属性发现走回退链，全缺 fail-loud。


## 添加能力

**新插件节点**（三站，缺一即断）：① 按域建模块（`packages/<data|model>/mflowy/<pkg>/<能力族>/`），入口加 `@node(namespace=...)`（归组）或 `@node()`（非归组）、纯变换边加 `@edge`（edges ns）；签名纪律：节点函数参数 **Annotated=节点自身参数**（`param_schema`——get_node/MCP info → YAML 模板面）；**非 Annotated=参数边**（`upstreams_schema`——接线上游**节点或边**函数的输出）；**边函数参数只能指向节点**（不吃边——边链成环无从发生），无 Annotated 面。② 同 PR 在域包 pyproject `[project.entry-points."mflowy.builtin_plugins"]` 手写声明——节点 `namespace.node_name`（非归组裸名）、边 `edges.node_name`，值 `模块路径:函数名`；③ `uv sync` 后 `list-nodes` 可见。判据：镜像路径有测试 + `list-nodes` 可见。

注册漂移不设镜像测试（2026-10 决策，限 `mflowy.builtin_plugins` 节点/边声明面——`mflowy.tools` 工具面两侧均沿用镜像测试）——CR 必查三项：**新 `@node`/`@edge` 必有声明、声明必有函数、entry point 名 namespace 段与装饰器 `namespace=` 逐字一致**；机器门在加载期按键查表（#140：未装饰/错 namespace/别名键统一「声明与注册不一致」fail-loud，翻 #126「只查装饰不比名」裁定），忘声明仍是静默面（目录无名即不可达），靠 CR + `list-nodes` 交叉检查。

**新 compute 工具**（JobProvider 委派，7 站点缺一即断）：

1. `packages/builtin_plugins/model/mflowy/model/job_provider/protocol.py` — 方法签名（`headers` 形参 = 请求 `_meta` 直传，runner 场景恒 None）
2. `packages/builtin_plugins/model/mflowy/model/job_provider/local.py` — 本地编排实现
3. 远程 provider 实现（module:Class 自定义包）— build command + `_dispatch`
4. `packages/builtin_plugins/model/mflowy/model/tools.py` — 工具包装 + docstring（schema 由此编译）
5. `packages/builtin_plugins/model/pyproject.toml` `[project.entry-points."mflowy.tools"]` — 静态注册条目
6. `packages/builtin_plugins/model/mflowy/model/templates.py` + `templates/*.yaml.j2` — 模板
7. `packages/builtin_plugins/model/tests/` — headers 契约 + provider 用例；工具清单从 `JobProvider` 协议自省生成（单源）

判据：`tools/list` 出现新工具且镜像/契约测试通过。本地分析类工具走 4/5 两站（server 经 entry points 自动注册）。

**新平台工具**（两站，缺一即断）：① `packages/mcp/mflowy/mcp/tools.py` 公有函数（Annotated 签名 + docstring 即 schema；非工具 helper 必须 `_` 前缀——模块公有面即函数面）；② 同 PR 在 `packages/mcp/pyproject.toml [project.entry-points."mflowy.tools"]` 静态声明 + `uv sync`。判据：镜像契约 `packages/mcp/tests/mcp/test_platform_tools_registry.py` 绿（声明面 ↔ 函数面双向差集）。

**第三方插件包**（活参考 = `packages/builtin_plugins/data/` + `packages/builtin_plugins/model/`）：以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]`（`namespace.node` / 裸 `node`），随 `uv --with` / 镜像 `MFLOWY_EXTRA_MODULES` 安装即生效。

## 文档模型

三类 markdown：根级治理文件、组件文档（`packages/<pkg>/README.md`，随包分发）、`docs/` 跨组件专题。路由（触发词打头）：

| 触发场景                   | 去处                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------- |
| 查术语 / 选词              | [CONTEXT.md](CONTEXT.md)（元词汇 + 禁用别名；按「引擎内核（共享）+ 三子系统」分组）    |
| 动 driver / YAML / 写插件  | [packages/driver/README.md](packages/driver/README.md)                                 |
| 查能力目录 / 边函数 / 遥测 | [packages/builtin_plugins/data/README.md](packages/builtin_plugins/data/README.md) + [packages/builtin_plugins/model/README.md](packages/builtin_plugins/model/README.md) |
| 写远程 provider（JobProvider 协议/自定义执行环境） | [packages/builtin_plugins/model/README.md](packages/builtin_plugins/model/README.md)（预测模型训练子系统——协议与实现住 model 域包） |
| 平台登录 / 遥测同意门      | [packages/mcp/README.md](packages/mcp/README.md)                                       |
| 用户方法论 / 案例叙事      | [docs/research-flow.md](docs/research-flow.md)（五律总纲 + 五幕叙事入口）              |
| 查数据清洗方法论（先验 / 检测算子 / 清洗算子矩阵） | [docs/method/data-cleaning.md](docs/method/data-cleaning.md)（研究领域知识，零词表绑定） |
| 查建模方法论（模型矩阵 / CV-HPO / 特征工程 / 评估） | [docs/method/model-training.md](docs/method/model-training.md)（研究领域知识，零词表绑定） |
| 查系统闭环设计（PILOT 三子系统 / 学习门禁 / 先验继承） | [docs/pilot-loop.md](docs/pilot-loop.md)（目标态叙事 + 落位现状对照表；BO 决策图 #171） |
| 规划新功能 / 查路线图      | [docs/roadmap.md](docs/roadmap.md)（已实现清单以 `tools/list` 为真源，roadmap 只留规划） |

**防漂移**：代码已表达的只链接不复述（文件树/签名/行号锚点留给代码）；只写追加式文档（新案例 → `examples/` 快照，新版本 → `CHANGELOG.md`）；"漂移可接受"仅限用户文档，release 边界回填——本文件与 skills 除外（见同步规则）。**剧本引用面谓词**（#65 裁定）：操作 skill（`.agents/skills/*`）参考节只允许引用工具名与入仓文档——引用实现路径（含包结构迁移前的旧路径）即 P0 漂移，删除而非修正。新建文档先过漂移测试："这文件里哪句话会因下次代码 PR 失效？"有答案就不建。

## Issue Tracker（gh CLI）

repo 从 `git remote -v` 推断。issue/PR 共号段：`#42` 先 `gh pr view 42` 再 `gh issue view 42`；外部 PR 不走 triage。

**Triage 五标签**：`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`。

**Wayfinding**（`/wayfinder` 用）：map = `wayfinder:map` label issue；child = sub-issue（降级：map body task list + `Part of #<map>`）；blocking = 原生 dependencies（降级：`Blocked by: #<n>` 行）；claim = `--add-assignee @me`；resolve = comment + close + 回写 map 的 Decisions-so-far。

## 词表

两层：**元词汇**（引擎内核共享词——节点/边/组/Context/会话——与三子系统域词的定义与禁用别名，按「引擎内核（共享）+ 三子系统」分组；新域词落所属子系统节，跨子系统共享的机制词才进内核节）见 [CONTEXT.md](CONTEXT.md)；**节点名唯一词表 = 运行期插件目录**（`driver/discover.py` 读两组 entry points；边注册于 `edges` ns 但不进 `list_nodes`）。解析三步（#137/#140）：注册表暖命中 → 目录查表 → 首载加载后按同一键查注册表（发现表键 = 注册表键，load 后 miss = 「声明与注册不一致」fail-loud）——解析面 ⊇ 词表（已导入的装饰节点可暖命中，生产不可达：一文件一节点下忘声明即无人 import）；暖驻留让渡卸载/后组覆盖的即时可见。**编译即校验**：Builder 增量编译（平铺有序列表 + 节点表）对参数边全量 fail-loud——拓扑序/词表 miss/撞名/类型门/参数 schema 校验（#131：已知键子集对装饰期模型，与运行期 validate_call 同源、未知键放过）。issue 标题 / 提案 / 测试名一律用 node/namespace 术语；目录里没有的概念 = 语言超纲信号。插件 SDK = `driver/node.py`（@node 装饰器）+ `edge.py` + `builtin_middleware.py` 尾链，破坏性变更受 CHANGELOG 语义化版本约束。
