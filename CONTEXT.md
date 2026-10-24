# MFlowy

YAML 配置驱动的模块化 ML 工作流引擎：能力以插件节点存在、图以 YAML 组装、全部能力经 MCP 工具暴露。本文件是**引擎元词汇**的词表——只定义这些词本身与禁用别名；节点与边函数的名字以运行期插件目录为唯一真源（`list-nodes` 可查），方法论术语见 [docs/research-flow.md](docs/research-flow.md) 自带术语表，均不在此复述。

## Language

### 单元与身份

**节点（node）**:
词表的一个目录项：一个被 `@handler` 标记的 pyfunc，身份是 entry point name。用户视角的"能力"即节点。
_Avoid_: step、模块（第一代退役词）

**namespace（组）**:
节点的归组前缀。参数边寻址时整组可命中（组寻址：全图命中组则只认组）。
_Avoid_: step 族、能力族

**handler**:
节点的实现视图——装饰器织入中间件链后的调度链函数。谈目录与图用"节点"，谈代码与调度链用"handler"。

**边函数（edge）**:
注册于 `edges` ns 的具名单向数据变换。无图位置，借消费方上下文晚绑定。
_Avoid_: 注入器、injector、getter（第一代退役词）

**插件（plugin）**:
分发单元：声明 mflowy entry points 的包，安装即注册、卸载即退出。内置能力包与第三方插件同构。

**工具（tool）**:
域包自持的编排入口（pyproject 注册表声明进 entry points 组 ``mflowy.tools``），host
（server/runner）发现并包装。工具组合节点，工具不是节点；平台工具（词表/mlflow 观测/
工件读取）住 host 自带注册。

**目录（词表）**:
运行期全部节点与边函数的枚举，机器可查、构建期校验。两词同义："目录"强调查询面，"词表"强调语言约束面。

### 寻址与图

**节点名三形**:
同一节点的三种写法：entry point 名 `plot.shap_summary`（目录与 YAML）、寻址名（node_key）`plot__shap_summary`（消费方参数边按此命中；`__` 连接符——组名与节点名皆可含 `_`，连接处无歧义，构造单一来源 `config.node_key`）、组内名 `shap_summary`（namespace 内的 node_name）。非归组节点（如 `predict`）三形合一为裸名。

**参数边**:
消费方签名上声明的数据依赖通道（裸类型命名参数），编译期按"边名 > 组名索引 > node_key 索引"三层查节点表解析（收集保序取全 / 单值取末位）。"边"单独出现有歧义（另有边函数），正式表述用全称。

**flow**:
YAML 的有序节点列表键（平铺**键控**叙事，#59 起）：列表项为单键映射 `{node_key: 配置体}`（键即身份，配置体仅 name/params/disabled），拓扑序强制，边完全由签名解析，YAML 零边声明。整份 YAML 文件称"图"。
_Avoid_: steps（第一代退役词）、嵌套叙事（flow/branches 树已退役）、namespace/node_name/enabled 配置体键（身份已上移为列表项的键）

**血缘**:
参数边名字解析命中的真实数据依赖。结构=血缘（#59 起）——prevs/nexts 即数据边，图物化即真图。

**Conf / NodeConf**:
Conf 是一份图的根配置，NodeConf 是图中单个节点的配置。两者都是真实类名，只差一个字。

### 执行与观测

**Context**:
DAG 任务节点：持有配置、结果与双向依赖边，是执行与记账的单位。
_Avoid_: 环境容器、执行上下文、DI 容器

**计划（plan）**:
构建期静态解析产出的边表与缺失清单，memoized 于所属 DAG 实例。执行只消费计划，不做名字解析。

**拉起（pull）**:
绑定期发现前驱未执行即当场执行。多个消费者拉同一上游只执行一次。

**血缘（lineage）**:
被消费的直接上游 run_id 集合（数据血缘而非路径），落 MLflow tag。成败 run 皆落盘。

**遥测面 / 数据面**:
中间件的两平面：遥测面是 `log_*` 领域观测（活跃 mlflow run 窗口内执行）；数据面是参数边绑定（链启动前、run 窗口外）。数据面已边函数化，不经中间件。

### 会话与域（0.2.0）

**域包（domain package）**:
按 MDP 域切分的能力包：`mflowy-data`（load/statistic/detect/clean + 清洗会话）、
`mflowy-model`（x_y/cv/x_transformer/model + 建模会话——cv 族 = cv.split 参数构建 /
cv.load 引用加载，下游经 `cv` namespace 参数消费）。依赖单向
data ← model；安装形态即域选择（`mflowy[data]` / `mflowy[model]`）。两包住分组目录
`packages/builtin_plugins/`（与 entry points 组 `mflowy.builtin_plugins` 对齐），
distribution 名与 extras 不含分组路径。
_Avoid_: builtin_plugins 指代**单个能力包**（1.x 单发行版结构退役——域包仍称域包，
builtin_plugins 只作分组目录名与 entry points 组名使用）

**工具注册（mflowy.tools）**:
域包工具的 entry point 声明通道：工具 = 域包公开函数（Annotated 签名 + docstring 即
schema），host（server/runner）发现并包装。镜像测试断言 entry points ≡ 工具模块公有函数。

**会话（session）**:
MDP 的环境面载体：转移/观测/奖励/预算在域包，π 在调用方（LLM）。清洗会话与建模会话
各持版本树与决策日志 (s,a,r,s′)，终态导出可全量重放的计划。协议/先验在 episode 内冻结。
回合步数（`Observation.steps`/`max_steps`）是会话预算计数，与 flow 词条的退役词
"steps"（YAML 控制结构旧称）不同物。
_Avoid_: 环境引擎（与 runner 混淆）

**先验卡（DomainCard）/ 协议卡（CVCard）**:
两个 MDP 的冻结输入：先验卡 = 数据域先验（target/id_columns/单位/范围/protected，survey
问卷经 compile_survey 编译）；协议卡 = 建模域对局规则（task/metric/CV/预算/约束，
model-brief 问卷编译）。协议卡的引擎入参形态（#66）：cv_card = cv.split 配置 yaml
明文（划分构建参数）+ start 显式参数（target/task/metric/candidates/max_steps/
id_columns——显式 > card 背景分层缺省），CVCard.build 仍是四类校验的单一事实源。
命名按内容不按过程；换协议 = board 作废 + test 重触，只能 abort 开新 episode。
_Avoid_: ExperimentCard、ModelBrief、ModelingCard（协议卡禁用别名）

### 规则与出题（0.2.0）

**规则（rule）**:
挂在被提议节点上的出题知识：`node.rule = {kind: fn}`（与 clean 域 `.repair` 同构），
声明"什么 findings/会话态下我是好的一步"。规则是节点附加属性而非词表项——无独立
entry point，枚举即对已加载节点 getattr。
_Avoid_: 以「规则引擎」裸指 RETE/DSL 类推理引擎（明确非需求——本仓自研件全称「规则引擎 SDK」）；rules. 前缀注册桶（#37 旧稿已废弃）

**资格门（gate）**:
规则点火前的会话态前置谓词（`(ctx) -> bool`，显式具名）。未过则候选恒不出现——
资格与权重互不代偿：S 无模型信息时 FE 候选恒零，不靠权重压序。
_Avoid_: 用高 weight 实现阶段前置（overrides 可改权重、不可越资格）

**记分牌（weight × severity）**:
候选排序面：weight = 域先验（委托人 overrides 可覆盖），severity = 触发 finding 的证据
强度（探索题记 0——无证据 = 零证据强度）；同分按 (owner, kind) 字典序保证菜单确定性。
