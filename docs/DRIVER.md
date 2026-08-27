# Driver — 轻量 DAG 编译与调度内核

`packages/driver/src/mflowy/driver/` 的架构文档。事实以代码为准，本文只沉淀心智模型、设计意图与赌注。

## 定位

**一个轻量的、能力自主注册的、由 YAML 编排的 DAG 编译与调度内核。**

职责一句话：构建期把 YAML 图（多为模板或 LLM 生成）编译成 Context DAG，运行期串行调度能力目录中的 handler——计算发生在 handler 内，引擎只组合与调度，并把每次运行装配成可追溯的实验记录（WorkflowResult + MLflow runs）。远程执行由 JobProvider 把整个工具抬到远端，driver 本体始终进程内。

四个目标词各自排除了一类方案：

| 目标 | 排除项 |
|------|--------|
| 轻量 | 事件总线、生命周期状态机、反射式依赖注入、热重载 |
| 自主注册 | 手工登记表、中心化 wiring、为注册写配置 |
| YAML 编排 | Python 代码编排、图 DSL、GUI 建图 |
| DAG 执行 | 通用插件框架、长驻服务运行时、动态装卸 |

**非目标**（范围选择，非缺陷）：并行执行——调度器（`workflow.py` LIFO 拓扑序）刻意串行；能力动态装卸——`discover` 目录在进程内 `@cache` 单向物化，不支持运行期装卸。

## 设计哲学

一句话：**图是从能力目录组装出的一等工件，让组装图的人（或 agent）免费获得追踪与词表约束。**

1. **图住配置，代码住目录**。能力（`@handler` 目录项）与图（YAML 工件）分离，派生两类角色：能力作者在词表内添目录项，图组装者（人、模板、LLM）拼 YAML。两类演化互不阻塞，driver 本质是两个角色之间的编译器。
2. **身份在 entry point name，不在代码**。节点地址是 `step.module`（entry point name）而非函数名或全局 ID：词表 = 运行期插件目录（构建期生成内置 + 安装期收集第三方），可枚举、可替换、可审计、可被外部扩展——LLM 与人共享同一图语言的前提（AGENTS.md 词表规则的动机）。
3. **边按类型寻址**。`prev(step)` 按能力检索最近的同类型上游，而非点名引用具体节点——换 module 不改下游连线，ML 实验的核心诉求。代价是边到运行期才解析：血缘 tag 是运行期补偿，参数校验延迟（「设计赌注」#3）是同一代价的另一面。
4. **表达力挂编译期，内核冻结**。领域定制全部走构建期变换（placeholder 简化、BuilderOption、Jinja2 渲染），调度内核保持纯调度，不加状态机、事件总线、生命周期。
5. **DAG 是实验计划书，不是计算调度计划**。串行调度、无增量跳过：每次运行必留痕，复用旧结果是显式 `run_id` 引用而非静默缓存；计算环境整体委派 JobProvider，图不感知。观测织入内核是这一立场的直接结果（代价见「设计赌注」#1）。

命题 1/4 合称"编译器前端"：全部组合发生在执行前，运行期只剩纯调度。

## 一次执行的生命周期

```mermaid
sequenceDiagram
    participant User
    participant Builder
    participant Conf
    participant Context
    participant Workflow

    User->>Builder: 提供 YAML（多为模板 + env）
    Builder->>Builder: Jinja2 渲染（变量替换）
    Builder->>Conf: 解析为 StepConf<br/>（递归 dataclass + 参数类型转换）
    Builder->>Builder: 应用 BuilderOption（解析期）
    Conf->>Conf: validate（build 时）<br/>占位符简化
    Builder->>Context: 前序遍历构建 DAG<br/>（steps 串行 / branches 并行）
    Context->>Workflow: 传递起点任务
    Workflow->>Workflow: Kahn 拓扑排序 + LIFO 调度
    Workflow->>User: 返回 WorkflowResult
```

| 阶段 | 代码 | 说明 |
|------|------|------|
| 模板渲染 | `builder.py: _load_yaml` | `{{ var }}` 注入 `env` 参数；生产入口多为模板而非手写 YAML（见「模板与片段组合」） |
| 配置解析 | `config.py: StepConf.__post_init__` | 递归 dataclass 化，同时触发按模块注册的参数类型转换 |
| 配置变换 | `builder_options.py` | BuilderOption 逐 step 应用，发生在 `Builder.__init__` 解析期 |
| 验证/简化 | `config.py: validate` | `build()` 触发；空占位删除、单层嵌套提升 |
| DAG 构建 | `builder.py: _build_tasks` | 前序遍历，串行 `steps` 链接、并行 `branches` 分叉；模块存在性在此校验 |
| 调度执行 | `workflow.py: run` | 见下 |

运行期语义：

- **Kahn 入度 + LIFO 就绪栈**：深度优先，刚解锁的下游先跑、独立分支按声明逆序。总时间不变（串行求和），但下游结果更早产出。
- **通道边界**：节点 `print`（stdout）被 `capture_prints` 捕获进 `NodeResult.output`，`logger`（stderr）走过程诊断（AGENTS.md 通道边界规则）。
- **失败语义**：任一节点异常即整图中止，`WorkflowResult.status=failed`、error 定位到 `step.module.name`；无重试、无续跑。
- **结果结构**：实验名/ID + 逐节点 run_id/状态/输出 + graph 三视图（name/tree/mermaid，默认 mermaid）。

## 核心机制

### 自主注册（最核心的设计）

新节点 = 一个 pyfunc + 一行装饰器，四处自动生效：

```python
@handler(inject_df, df_diff)   # packages/builtin_plugins/src/mflowy/builtin_plugins/cleaners/filter/common_filter.py
def common_filter(...) -> pd.DataFrame: ...
```

1. **身份声明**：`hatch_metadata.py` 构建期 AST 扫描 `@handler` 函数生成 `[project.entry-points."mflowy.builtin_plugins"]`——name 即身份 `step.module`（目录→step 映射在 `_STEP_OF_DIR`，新能力族漏配构建即报错）
2. **能力标记**：`handler.py: @handler` 织入中间件链后把 `.handler`（调度链）与 `.convert_params`（参数转换器）挂到函数属性上，装饰器返回原函数（直调不破）；`mlflow_log` + `stop_on_error` 强制尾链，每个 handler 必在 mlflow run 内执行
3. **目录与解析**：`discover.py` 按序读两组 entry points（`mflowy.builtin_plugins` 内置 + `mflowy.plugins` 第三方，后组覆盖前组并告警），`discover()/has()/list_all()` 纯元数据零 import；`_load_fn` 惰性 import 并以双属性校验声明一致性，坏声明 fail-loud
4. **用户文档**：`module.py: get_module_info` 内省 `Annotated[T, "描述"]` → MCP `tools/list` schema；参数转换器（`_build_params_converter`）从同一签名构建 YAML 值 → 类型实例（ContinuousSpace/DiscreteSpace/TASKTYPE）

**第三方插件**：以 mflowy 为 base 依赖，在自身 pyproject 声明 `[project.entry-points."mflowy.plugins"]` 即可（`uv --with pkg` / 镜像 `--build-arg MFLOWY_EXTRA_MODULES` 均随安装自动生效）。扩展单元 = step + 模块 + 注入器：新 step 的作者需自带 `Get*/inject*` 中间件对（`middlewares/data_inject.py` 是范本），否则模块拿不到上游数据。

**editable 陷阱**：entry points 元数据在 `uv sync` 时生成，新增 compute 模块后必须重跑 `uv sync` 才会在 `list_modules` 出现。

### 类型注解单源

`Annotated[T, "描述"]` 是全引擎的契约单一事实源，四处消费（见上）。签名约定：裸类型参数 = 中间件注入（如 `df`），`Annotated` 参数 = 用户可配（`get_module_info` 据此过滤）。

### Context = 数据节点（注意命名）

`Context` **不是**依赖注入意义上的"环境容器"——它是 DAG 任务节点：持有 `conf`、`result`、双向依赖边（`_prevs` 溯源 / `_nexts` 调度）。

数据依赖经 `ctx.prev(step)` 声明：BFS 向上游搜索指定类型的最近前置（`max_depth=20` 兜底），命中即消费其 `result`，**并写血缘 tag**（`context.py: _PATH_TAG_KEY = "mflowy.input_steps"`）——副作用不可撤销但全程可追溯。

`required=True` 时找不到前置抛 `PreviousContextNotFoundError`；`ContextVar` 计数器保证并发 `Builder.build()` 编号隔离。

### 中间件链

注册时一次性织入：`@handler(mw1, mw2)` → `mw1 → mw2 → mlflow_log → stop_on_error → handler`。中间件两处：内核默认尾链在 `driver/builtin_middleware.py`；注入器与领域日志在 `packages/builtin_plugins/src/mflowy/builtin_plugins/middlewares/`（`Get*` 数据访问 + `inject_*` 注入 + `log_*` 观测）。

### 模板与片段组合

生产入口几乎不是手写 YAML，而是 **Jinja2 模板（`packages/mcp/src/mflowy/mcp/templates/`）+ `env` 注入**：每个 MCP 工具对应一张模板，用户参数与 `modeling_steps` 片段渲染成完整配置。

片段可再加工后跨工具复用：`explanation` 工具复用 `modeling` 的 `modeling_steps_yaml` 片段——Builder 解析 → `prune_model_step` 改写（model 步替换为 loader）→ `serializer.steps_to_yaml` 重新序列化 → 注入另一张模板（`mcp/job_provider/local.py`）。`serializer.py: _plain`（Enum→名、dataclass→dict）与参数转换器构成 **YAML 往返闭环**，配置因此是可生成、可改写、可再注入的一等公民。

### BuilderOption：StepConf → StepConf

- **工厂期与闭包期分离**：`prune_model_step` / `resume_model_step` 创建 option 时一次性完成 model 参数解析 + MLflow 查询（构造 `{module: run_id}` 映射），返回的闭包只做查表改写——应用期无外部调用。
- **语义分工**：`prune_model_step` 未命中即剪枝、`resume_model_step` 未命中保持原状继续训练——覆盖"复用旧实验"的两种形态。
- **结构剪枝不算 Option**：`prune_x_transformer_step` 需看 nexts 才能判定下游是否消费 transformer，故内联在 `_parse_step_dicts`。
- **只编译不运行**：`builder.build(preview=...)` 不调 `run()`，validate 类工具借此做编译检查（模块存在性校验 + mermaid 预览）。

## 设计赌注与已知张力

1. **MLflow 深耦合**：`workflow.py`（setup/run）、`context.py`（血缘 tag）、`handler.py`（默认尾链）三处直连观测层——"轻量引擎"与"MLflow 唯一观测层"叠加的结果（哲学命题 5 的代价）。若未来抽独立引擎，`context.py` 血缘写点是唯一深的缝。
2. ~~**注册静默丢失**~~ **已解决**：发现从 import-all 扫描改为 entry points 元数据——目录查询零 import（缺 extra 的模块照常列出，`get_module_info` 时才暴露不可导入），加载才 import 且坏声明（未标注 `@handler`）直接报错，构建期另有 `hatch_metadata.py` 的 AST 扫描双重把关。
3. **参数校验时机**：参数名错误要到执行到该节点才 TypeError；类型转换已在解析期。解析期 keys 比对是候选增强。
4. **`module.py` 的 `eval(annotation)`**：解析字符串注解，仅处理本进程注册的函数；换 `typing.get_type_hints` 可收敛。

## 扩展点速查

| 需求 | 动作 |
|------|------|
| 新节点（插件） | `packages/builtin_plugins/src/mflowy/builtin_plugins/<step>/**` 建 `.py` + `@handler(...)`，零配置，`uv sync` 后生效 |
| 新能力族（新 step） | `hatch_metadata.py: _STEP_OF_DIR` 加目录→step 映射 + 为该族写 `Get*/inject*` 注入器对（`builtin_plugins/middlewares/getters.py` 范本） |
| 第三方插件 | 以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]`，name 格式 `step.module` |
| 修改/剪枝配置 | `builder_options.py` 写 `BuilderOption`，`Builder(..., opt)` 传入 |
| 新横切关注点 | builtin_plugins `middlewares/` 建 `log_*` 或功能中间件，装饰器引用 |
| 查询已注册能力 | `discover.py: list_all()/has()` / MCP info 工具（list_modules / get_module_info） |
| 复用/改写另一工具的 steps 片段 | Builder 解析 + BuilderOption 改写 + `serializer.steps_to_yaml` 再注入（见「模板与片段组合」） |
| DAG 可视化 | `Workflow.__repr__` 三视图（name/tree/mermaid），mermaid 为默认 |

## 相关文档

- 远程执行（引擎之上的 JobProvider 层）：[REMOTE_MODELING.md](REMOTE_MODELING.md)
- 引擎消费方：`packages/mcp/src/mflowy/mcp/tools.py` · `.../mcp/templates/` · `.../mcp/job_provider/protocol.py`
