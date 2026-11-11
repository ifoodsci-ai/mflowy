# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 远程执行切片③：ensure_platform_experiment 实验就绪工具（#196，feat）

remote 系列工具面第一件（#156/#159 决议）：查建平台实验 + 文件清单增量同步 +
可继承面回报——后续 compute 提交的前置动作（CONTEXT 词条「实验就绪」）。

- 新平台工具 `ensure_platform_experiment`（mcp 两站注册）：传 id = 查询模式
  （校验存在，404 fail-loud **绝不静默转创建**）；不传 = 创建模式（name 必填）；
  id 与 name/desc 同传 = 参数来源冲突拒绝。返回
  `{platform_experiment_id, files: {uploaded/updated/skipped: [{local, remote}]},
  inheritable: [{job_id, status}]}`（inheritable = outputs 下有 `{job_id}.zip` 的
  job，newest first、status 不过滤；发现面取代已砍除的 list_platform_jobs）。
- 增量同步：本地 MD5 ↔ 远端 etag 裸字符串全等——「裸」= 剥 S3 引号（网关恒带引号
  下发，不剥则幂等面失效，S2 E2E 实证；multipart etag 含 `-` 自然落重传）；size 不
  参与判定也不进返回面（#156 返回 shape 为准，agent 对本地文件自知）；不删除远端
  未列文件；清单预检 fail-loud 家族（dotfile 拒整次 / 目录 / 撞名 / 不存在 / MCP
  相对路径拒收）。
- 聚合面：根 `[mcp]` extra 增随 `mflowy-model`（#159 §5「既有 model 依赖传递」前提
  在包级不成立——mcp 包保持零域依赖，server 分发形态经聚合取得 model；裸 mflowy-mcp
  嵌入形态按「安装形态即通道」fail-loud ImportError）。
- **路径形态通道纪律落地**（#159 §6）：`route_task_dir` 增 `channel` 入参
  （runner="cli" / server="mcp"）+ 共享解析助手 `resolve_input_path`——相对路径
  CLI 按 cwd 接受、MCP 拒收 fail-loud（server 无业务 cwd 语义）。
- **MCP 通道 fail-loud 消息透传**（server `_fail_loud`）：SDK 缺省把工具异常掩成
  `Error executing tool <name>`（多租户防泄漏缺省）——统一转 `ToolError` 透传
  （消息带异常类名，CLI 通道本就全量打印；红线 1 的候选清单文案必须达 agent）。
  E2E 首跑实证：透传前负向断言全被掩。
- 懒加载边界：工具体内 import model 包 platform_client（mcp 保持零域依赖 host——
  注册无条件、执行随安装形态，裸 host 无 model 时 fail-loud ImportError，零降级机器）。
- S2 工具面 E2E 驱动入仓（`make e2e-remote-tools` → `scripts/_e2e_remote_tools.py`，
  MCP stdio 真实通道 + 遥测 off 常设）；本轮 **13/13 绿**（含 upload-url 对 CLI
  token 实测 200——#161 期 files 子路 scope 疑虑解除）。J1 继承面断言随切片④生长。

### 远程执行切片②：platform_client + 错误族（#195，feat）

RemoteExperimentClient 落地（`job_provider/platform_client.py`，#157 决议）——系列
后续切片（ensure 工具 / RemoteJobProvider / download 工具）的共同地基，纯单测面。

- 九方法面（`list_jobs` 砍除）：实验查/建、presigned upload/download-url、list_files
  （服务端目录树客户端拍平）、create/get_job、job logs、list_presets——wire = pydantic
  `extra=ignore`，时间戳 int64 epoch 秒原样（`workspace_size_mi: int | None` 承接
  上游 #45 regen 缺失窗口）。
- 鉴权接线：逐请求 `ensure_access_token`；401 → force 强刷一次重放一次，仍 401 →
  `LoginRequiredError`。`ensure_access_token` 增 `force` 钮（utils API 变更：未过期
  也走锁内刷新——服务端判无效时本地 exp 不可信；等锁期间并发轮换则复用新值不空刷）。
- 错误族：`PlatformApiError`（endpoint + base_url + 三形态提取的 message；传输层
  异常统一归此，含 MFLOWY_BASE_URL/NodePort 可达性提示——S8）/ `PlatformNotFoundError`
  （404，ensure「绝不静默转创建」消费）；presigned 传输段（`upload_presigned` /
  `download_presigned`，tmp+rename 原子写）错误面剥 query、只报 status + S3 `<Code>`，
  **凭据不变量**（#186）：任何报错不含 X-Amz 签名材料（契约测试钉住）。
- base_url 解析链公开单源 `resolve_base_url`（与 auth 同链同缺省，不另立 PLATFORM var）。
- `JOB_TERMINAL_STATUSES` 常量单源（logs 工具 terminal 判定，切片④消费）。
- mount_type 字面量校验 fail-loud 附候选；404 候选清单纪律（附已查 id + 建议动作）。
  契约测试 = MockTransport 全桩零真实网络：client 17 例（三形态错误提取参数化 ×3，
  凭据不变量覆盖 S3 错误分支与传输分支）+ auth force 镜像两例。

### 远程执行基座：镜像 zip 往返 + 版本钉 + 依赖面（#153 切片①，feat）

RemoteJobProvider 接入上游实验服务（决策地图 #153）实现系列的首个切片——镜像与依赖
基座；S0/S1 平台契约回归本轮全绿（26 断言 / 0 失败 / 1 过渡 warn，本树重建镜像
`training-job:0.2.0`，证据随 PR 描述留档）。

- `docker/Dockerfile` 运行时层新增 `zip`/`unzip`（远程任务产物打包与跨 Job 继承解包，
  任务包装器与调用方参考模式共用）；烧平台 Job 容器契约 ENV：
  `MFLOWY_TASK_DIR=/user-data/workspace`、`MLFLOW_TRACKING_URI=sqlite:////user-data/workspace/mlflow.db`
  （显式 `-e` 覆盖优先，本地/MCP 部署语义不变）；`HOME=/tmp`——nobody 缺省 HOME 不可写，
  而 `utils/mlflow.py` setup() 急切 mkdir `$HOME/.mflowy`（#161 研究回填）。
- 镜像 tag 版本钉（#187 统一版本钉）：Makefile `IMAGE = training-job:$(WHL_VERSION)`
  （pyproject 版本单源；提取件 `scripts/_whl_version.sh` 与 e2e 驱动共享），`latest`
  全面退役；dead 变量示例（`IMAGE_TAG`/`IMAGE_REGISTRY`）从 help 清除，整串覆盖走
  命令行 `IMAGE=`。
- 依赖面（#157/#159 决议：`[remote]` extra 取消）：`httpx2>=2.12` 与 `pydantic>=2.13.4`
  进 mflowy-model 默认依赖——wire 模型与 JobOutcome 直用 pydantic（不再靠 driver 传递
  侥幸，红线 6 清偿）、HTTP 栈 = httpx2（锁内既有，MCP server 形态零新增包）；后续四新
  平台工具无条件注册、零降级面。锁文件仅增依赖边，解析版本零变动。
- E2E 验收 runbook 入仓（#165）：`docs/research/2026-11-11-e2e-remote-runbook.md`——
  remote 实现系列 PR 的合并门禁（S0–S8 九段 + 切片→必过映射）。S0/S1 平台面驱动
  `scripts/_e2e_remote_platform.sh`（OTP 登录 → preset 断言 → 建实验 → 上传 →
  CreateJob cmd=sh → 轮询断言；`make e2e-remote`；redis 凭据经 REDISCLI_AUTH 注入、
  网关调用统一 30s 超时）与基座载荷（`scripts/_e2e_remote_task.sh` /
  `_e2e_remote_payload.json`）同批入仓；roadmap 增发版轮镜像策略行。
- `CONTEXT.md` 预测模型训练子系统落远程执行十词（#160 终审原样搬运：平台实验 / 追踪
  实验 / 平台任务 / 执行模式 / 模式专属参数 / 受理面 / 继承 / 可继承 zip / 继承源
  任务 / 实验就绪），占位随本 PR 消解。

### 系统设计文档：PILOT 入仓 + 三子系统重排（#171 / #185，docs）

- 新增 `docs/pilot-loop.md`：PILOT-AI4S（先验继承闭环优化测试系统）系统设计 canon——
  数据清洗 / 预测模型训练 / 贝叶斯优化推荐三子系统 × 先验继承 × 学习门禁；头部「落位
  现状对照表」随实现推进更新（正文不改写）。README 主题对齐闭环（PILOT 叙事 + 三子
  系统状态表）、roadmap 增「工业实验闭环（PILOT）」规划行（#171）。
- `CONTEXT.md` 重排为「引擎内核（共享）+ 三子系统」分组（#185）：跨子系统机制词
  （单元与身份 / 寻址与图 / 执行与观测 / 会话与域）归引擎内核节；先验卡（DomainCard）
  与「缺陷与判定」归**数据清洗子系统**；协议卡（CVCard）归**预测模型训练子系统**，
  并留远程执行十词（#160 终审）占位——随首个 remote 实现 PR 落该节（#153）；**贝叶
  斯优化推荐子系统**占位，随 bo 域落地 PR 填充（#171）。词条定义零改动，唯一内容
  变更 = 「先验卡 / 协议卡」合词条按域拆两条。
- **远程执行文档归预测模型训练子系统**（#185）：「JobProvider：接入自定义执行环境」
  节自 mcp README 迁入 model README（协议、实现与契约测试本就住 model 域包），悬空的
  JobProvisioner 规划行不再随迁（#153 裁定被设计取代）；mcp README 开头句 / 标题 /
  直接 import 行对齐「零域依赖 host」事实；AGENTS.md 路由行拆分（写远程 provider →
  model README），driver README / README（C4 与 distribution 表）/ CONTRIBUTING 的
  归属表述与链接同频。
- 同频：roadmap 能力域行名对齐三子系统全称；AGENTS.md 词表节与 README 文档节随
  CONTEXT.md 分组更新；mcp README 工具表补 POMDP 会话 / 报告素材 / YAML 预检组
  （全部 31 工具入表）。其余路由表文档（data README、research-flow、method 两份、
  pilot-loop 本体）逐份审查裁定不动。
- 存量漂移修复：`.mcp.json.example` / mcp README / driver README 退役 extra
  `[mcp,modeling]` → `[mcp,model]`（照文档安装即报错）；model README 已退役契约
  `Candidate` 残留；CONTRIBUTING 依赖归属段（`[stats]`/`[modeling]` 旧 extras）对齐
  域包形态。

### 用户文档拆解：方法论知识库两份参考 + 叙事瘦身（#163，docs）

- 新增 `docs/method/` 两份方法论参考：**数据清洗**（数据先验提示与示例 / 缺陷检测算子
  矩阵 / 清洗算子矩阵）与**模型训练**（ML/DL 模型矩阵 / CV-HPO / 特征工程 / 模型评估）。
  定位研究领域知识，**零词表绑定**（用户裁定：矩阵行 = 研究领域的算子/模型空间，不锚定
  引擎当前词表、不做一致性断言、零镜像测试）；近 70 条文献锚点逐条核验存在性（题名/
  作者/年份/venue，一手元数据页）并标注核验级别（摘要级/书目级），无裸引用。
- `docs/research-flow.md` 瘦身为叙事入口：五律、三道冻结门、五幕对话、术语表保留；文末
  锚点表瘦身为五律级（深度锚点归两份方法论文档，避免双份维护），导读处深链下钻路径。
- 同频修复存量漂移（叙事案例对齐当前动作面）：第一幕 ID 列语义（#112——特征剔除防泄
  漏，never_drop 只圈目标/受保护/时间）与建档卡注释；第二幕观测缺陷 kind 与算子名、
  #117 提交面双必填（finding_ids + reason）、`clean_finish(preview)` 探针；第三幕开局
  签名（file_path + cv_card）；第四幕八动作面（select_model/set_hp/set_hp_space/set_seed/
  select_FE/run_hpo/final_train_test——`final_train_test` 承接退役 `eval_full` 单向门）与
  `modeling_finish(pdf)` 收局剧本；术语表同步（选项面/光标/账本）。
- 修正文献误归属一处："Krishnan & Stonebraker, *CleanML* 2020" → Li et al., ICDE 2021
  （arXiv:1904.09483 comments "published in ICDE 2021" 一手核验）——更正记录随锚点入
  data-cleaning.md 文末。
- AGENTS.md 文档模型路由表 +2 行（两份方法论文档去处）。

### df_diff 值比较守卫单源——改名列不再崩整条流（#149，fix）

- 中间件 `log_df_diff._compute_diff` 的值比较守卫从「仅看 shape」收敛到与决策日志
  `_diff_summary` 同一谓词（新共享件 `types.frames_value_comparable`：形状 + 列标签 +
  行索引全等）——改名列算子（shape 不变、标签不齐）此前进 `DataFrame.compare` 直接抛
  ``identically-labeled`` 使整条 DAG 失败，现如实记 ``value_diffs=None`` +
  ``columns.removed/added`` 成对（改名 = 一删一增）。
- 词表冒烟补**编译链形态**（load__file + clean__op 两节点流 Builder 真跑，#149 验收）：
  算子 × 中间件交互缺陷第一时间暴露——直调 `node.fn` 的旧形态正是改名列崩溃长期
  未暴露的原因；fn 级用例保留（快，钉 INV-6/参数面闭包）。

### BREAKING：strip_units 退役——normalize_units 承接（#152）

- clean 词表 24 → 23：`clean.strip_units` 删除（模块 + entry point）。退役依据：三独立
  缺陷（`str.rstrip` 字符集剥离——unit=`'kg'` 时 `'500g'` 被当 500 kg；
  `errors="coerce"` 解析失败静默造 NaN；改列名 `列`→`列(单位)`——diff 值比较不可比
  （#149）、守卫把改名当删列、冻结卡旧列名先验失效）+ 编译上下文实际不可用（df_diff
  对改名列必崩，词表冒烟直调绕过中间件故长期未暴露）+ 仓内零消费方；单源收敛——
  恒等倍率形态 `normalize_units(columns={列: {to: u, factors: {u: 1}}})` 即其严格超集。
- 路由：format_anomaly → `{normalize_units, cast_cols, to_category}`；卡字段
  unit_hints 语义挂靠 normalize_units（单一单位列 to/factors 先验）。
- **迁移**：`clean__strip_units: {columns: {列: 单位}}` →
  `clean__normalize_units: {columns: {列: {to: 单位, factors: {单位: 1}}}}`；
  新算子**不改列名**（旧行为改名为 `列(单位)`）——按新列名取列的直接调用方需调整。
- 同频：SKILL 处置表 / cast_cols 文案 / README 词表 / research-flow 第一幕菜单示意。
- 附带修复（#151 缺口，由本票验收测暴露）：driver 遥测面 `_flatten_params` 叶键按
  mlflow 键名字符集确定性转义（集外字符与转义标记 ``-`` → ``-XX`` 十六进制，编码
  单射——转义段不与原生 ``-xx`` 键同形）——单位 token 键
  （``%``/``°C``）此前在任何 DAG 上下文执行必炸（mlflow 键名校验拒绝；中文键
  在 3.15 的 Unicode ``\w`` 下实为可收——转义按白名单统一，跨版本保守面）；
  业务参数面（NodeConf.params/plan.yaml）不受影响。

### 单位混录三层治理：mixed_units 缺陷类型 + normalize_units 算子（#151）

分工 = 检测只报事实、π 判量纲（引擎零单位知识）、算子只执行换算表。

- **检测**（`detect.stringly` 重构为 token 画像）：单位 token 支持多字符（≤8 字，
  超长按自由文本计不可解析），裸数字 token = `""`；同列 ≥2 种 token → 新缺陷类型
  **`mixed_units`**（DefectType 第 12 成员，0.x minor；恒 high、权重 0.15），证据 =
  完整 token 分布 + 不可解析值**形状签名**（连续数字→9、字母/中文→A、符号保留——
  `<0.01`→`<9.9`、`ND`→`A`、`12.3±0.5`→`9.9±9.9`，不泄露原文；替代遮蔽样例——
  mask_text 会把 `ND` 遮成 `**`，检测限/不确定度写法不可辨）；恰 1 种 token 仍
  `format_anomaly`（多字符单一单位列从**零 finding** 变可见——v1 单字符正则盲区）；
  token 大小写不归并（mL 毫升 ≠ ML 兆升——归并是判断，归 π）。
- **clean 词表 23 → 24**：`normalize_units`（π 给换算表 `factors = {token: 倍率}`；
  列名**不变**——目标单位留 params 单源；覆盖校验/不可解析值/factor 正数域/to 恒等律
  全 fail-loud；数值列走 `""` factor 统一尺度；仿射换算 °C→°F 走 python 逃生舱）；
  词法与检测单源 `split_number_unit`。路由：mixed_units → normalize_units（不提示
  strip——剥单位对混录静默算错）。
- 问卷 unit_hints 出题只对单一单位列（混录列由会话内 π 读分布后自定换算表）。
- SKILL 处置表增 mixed_units 行（同量纲换算 / 异量纲报告用户的量纲判断协议）。

### 清洗计划导出改版本树溯源——跨分支回滚后计划与交付数据对齐（#106，fix）

`_plan_yaml` 的「栈弹出」重建仅对沿当前栈回拨正确；#89 rollback 的指针回拨允许跳回
任意已 commit 版本（含弃分支），跳回后 finish 的 plan.yaml 与 final_path 脱节（计划
重放产物 ≠ 交付数据，跳回后新 apply 同样失真）。改为重放指针演化重建版本树（apply
的父 = 当时指针，路径 = 父路径 + 本步）——不落新日志键、无迁移面，旧会话日志天然
兼容；附指针重放守卫（重放终 ≠ 当前指针即 fail-loud，杜绝未知记录类型下的静默失真
计划）。验收 = 两条新契约测试真跑 Builder 重放并按 artifact sha256 对账（跳回弃分支
→finish；跳回→再 apply→finish）。

### 数据域对齐《数据质量 Agent Workflow 设计文档 v1.0》——决策地图 #110 落地（BREAKING·0.2.x 批次）

规则引擎时代结束：菜单/候选生成全线退役，π 可见面 = 动作词表全集 + 观测证据；
机制 = 事实，π = 选择。

- **BREAKING**（导入/工具面）：规则引擎 SDK `driver/rule.py` 整模块删除（Rule /
  RuleEngine / compose_should_stop / salience 组合子——两域菜面全灭，零存活消费者）；
  `Activation` 契约模型、`POMDPSessionHandle.propose`、`rules.pkl` 盘上件、
  `POMDPSession(*rules)` 构造签名随废；六操作换血 **propose 出局、`observe(sid)` 入局**
  （幂等只读零转移，终态照常可观测）——两域同步：`clean_propose`/`modeling_propose`
  工具退役，`clean_observe`/`modeling_observe` 上位；model 域 `propose_rules.py` 菜面
  删除（finding 词表单源 `taxonomy.py`；观测读时投影决策面——findings/combo_key/
  models 摘要/has_cat/face 候选超参面）。
- **BREAKING**（缺陷契约）：`kind` → `defect_type` 全面更名（DataError/KINDS 集中表
  退役）——`DefectType`/`Action`/`Severity` 三封闭 str-Enum 单处定义（12 缺陷成员 =
  文档核心值 ∪ 现状扩展值，pii 随检测器毕业）；改名五值：missing→`high_null`（纯门禁
  0.05 报告/0.30 high）、constant→`constant_col`、outlier→`outlier_numeric`、
  stringly_numeric→`format_anomaly`（子集语义）、id_like→`suspected_id`（打分检测器
  `detect.suspected_id` 四出口：composite 组合键 / 强先验直出 / 反先验冲突 / 常规打分
  ——新增 `suspected_composite_id`/`prior_conflict`）。**Finding 六字段**（finding_id
  结构键 `{defect_type}::{锚点}` + 序号后缀 / 定位键 column|rows 互斥（rows 字段退役
  改行级定位键）/ severity 三档（float 退役——score 进 rationale 前缀）/ evidence
  脱敏 str）；severity 分档罚分进质量分 Φ；权重随枚举全量显式覆盖（无兜底）。
- **BREAKING**（先验卡）：PriorKnowledge 七字段并入 DomainCard（`primary_key_columns`
  /`foreign_key_columns`/`composite_keys`/`known_non_id_columns`/`hinted_id_patterns`/
  `source`）；**never_drop 收缩为 target/protected/time——ID 列不再自动免删**（drop/
  hash/mask 归 π 决策，误删兜底 = POMDP 回滚）；model 域 id 剔除面 = known_id ∪ pk ∪
  composite 成员（fk 不剔）。
- **clean 词表 19 → 23 原子**：新增 `mask_cols`/`hash_cols`（匿名化——脱敏判据与检测
  证据同源 `mask_text`）/`cast_cols`/`to_category`（format_anomaly 修复面）；判据成文
  「值域封闭，身份开放」入 CONTEXT.md。
- **决策面换血**：`clean_apply` 增 `finding_ids`（非空校验 ⊆ 当前 findings，幻觉免费
  失败）+ `reason`（必填非空）；决策日志键 `action` → `clean_op` + 派生 `intent`
  （drop_cols⇒drop、其余⇒fix、finish 残留⇒keep）；no-op 步否决；`clean_finish` 增
  `keeps` keep 声明（residual 逐条附理由 + manifest = ID 先验 ∪ keep 派生）；观测
  `state_digest.problems` = Finding 六字段全量透传、`routing_hints` 常驻（提示非约束）、
  列画像 `missing_rate` 常驻。
- **问卷/naive 层**：id 问题单问两档证据（全唯一疑似 + 组合唯一疑似——univariate
  `hits["composite"]` 组合候选）；`defect_priority` 卡覆盖键按枚举值收（未知键
  fail-loud）。
- **测试面（#119）**：SMOKE_DF/SMOKE_CARD 行为闭环（产出并集 = DefectType 全集、
  severity 三档、score 前缀、脱敏、finding_id 结构键）；三面闭合（产出者 × 权重 ×
  路由提示）枚举驱动；clean 词表参数化冒烟（23 原子最小参数 + INV-6 返回新 df）。
- dry_run 维持退役（#65 钉死测试不拆）；rounds/phase 枚举不引入（相界 = 「谁能调
  什么」——算子不暴露为 π 工具即 API 面事实）；INV-3/INV-7 随载体死 dissolve。

### Changed（clean 剧本）：先疑记法、后判离群——成批越界/离群不再默认截断删行

range_violation / outlier_numeric 成批命中时先怀疑记法混录（同列两种记法差 10^k 倍：
小数 vs 百分数、g vs kg），决策指南默认档（`range_clip` / `cap_*` / `drop_outlier_*` /
`drop_outside`）暂停——证据样例已脱敏、倍率无法从观测读出，π 停下报告用户确认换算
规则，确认后经 `python` 逃生舱换算（reason 记规则与倍率）。缓解 RANGE_VIOLATION 路由
提示仅指向 `range_clip` 的破坏性走向；引擎侧单位归一见 #151，纯数字双记法见 #79。

### model 域化石退役：用户超参缺省迁冻结第四面（#141，#126 目标收官·BREAKING·model 域导入面）

- `model/utils.py` 活反射提取器（每次调用 `inspect.signature` 取超参默认值）删除；
  等价面 `family_param_defaults(entry)` = (`param_fields` 带默认 ∪ `upstreams_schema`
  可选边默认) − `MACHINERY_SKIP`——#138 冻结面过滤派生，五族键集与默认值本尊实测
  全等（对账钉测以活签名为独立 oracle 锁定等价；五族可选边项实测空集——union
  取全目录公式一般性：79 内置节点 28 例带非 skip 可选边默认）。**BREAKING**：
  ``mflowy.model.model.utils`` 化石函数删除（原跨模块导入面）。
- 消费位迁移：`make_machinery`（签名不变，内部经正规门面 `get_node` 按 node_key 取
  条目）；`family_face` 同改；`protocol._validate_hints` 改调 `family_face`（消除其
  绕过单源直查签名的重复）。反射台账（`tests/test_reflection_core.py`）model 域
  utils.py 位点清零——棘轮收窄。
- #126「签名解析单源收敛到注册时」目标就此收官（kernel 侧 #138 + model 域本票）。

### BREAKING：Node/Edge 装饰期反射核冻结 + fn 反射单源工具包（#138）

- **BREAKING**（导入面）：`driver/binding.py` → **`driver/reflect.py`**（fn 反射工具包——
  职责 = 反射处理 fn 拿到 param 面与 upstream 面；对被装饰 fn 的 typing/inspect 逻辑
  全仓单源）。非反射件拆居：`Upstream`/`UpstreamMeta`/`MISSING`/`_NOT_RUN` →
  `driver.context`（取值闭包执行面随唯一运行期消费者）、`raise_if_params_collide` →
  `driver.node`（M1 守卫随 Node 参数面）、`address_kind` → `driver.config`（与寻址名
  编解码同所）；`node._params_contract` → `reflect.params_contract`（返回三面：校验
  模型 / JSON Schema / fields）、`module._render_hint` → `reflect.render_hint`。
  `from mflowy.driver.binding import ...` 需改（仓内零残留）。
- **契约扩展**（`Node`/`Edge` Protocol）：装饰期一次冻结反射核——`upstreams_schema`
  （参数边四元组 `ParamEdge(name, hint, optional, default)`，必选 default 为 `...`）+
  `return_hint`；Node 另增 `param_fields`（Annotated 参数的 create_model 字段定义，
  只读映射——第四面 Python 形态，服务漏④后续票）。注册期解析模型：节点参数按
  Annotated 二分（参数边 → `upstreams_schema`，DAG 图校验面；节点自身参数 →
  `param_schema`，YAML 模板面）；边参数全为参数边。投影（via 展开/寻址 kind）依赖
  词表，查询期自冻结核派生（装饰期算投影 = import 顺序脆弱性，不做）。
- **生产活反射退役**：builder 直连绑定 / via 内层 / M1 名集读冻结核；module 词表投影
  （via 展开、default repr、expect 渲染）读冻结核；`Context.output_hint` ← `Node.return_hint`。
  **BREAKING**（签名）：`verify_output_hint(name, expect, output_hint)` 第三参由上游 fn
  改收预计算返回注解（节点/边两侧直传，内部零反射）；`param_edge_names` 入参由 fn
  改收冻结面（`upstreams_schema` / `upstream_params` 产物）——纯投影零反射，node 运行期
  防线与 builder 编译期共用同一 helper（名集投影单源）。
- **BREAKING**（装饰期语义）：`@edge` 守卫（红线 1）——边函数参数带 Annotated = 契约
  违例，装饰期 `TypeError` fail-loud（点名参数、不入注册表；现基线零命中）；边注解
  改为装饰期解析（与 `@node` 同构——@node 自装饰期派生 param_schema 起即如此）：
  注解须在装饰点可解析，模块内后置定义的前向引用此前可在编译期解析，现装饰期即
  `NameError` fail-loud（内置 5 边零命中）。
- 钉测：`tests/test_reflection_core.py`——反射原语单源收敛（AST 扫描 + 工具包外存量
  台账双向棘轮：mcp runner 工具面 1 + 漏④ model 域 3 反射调用点）、冻结点钉测（活反射
  成员只许 node/edge 装饰期冻结点导入——与收敛钉测合取 = 消费路径零活反射，静态判定
  零替身）、全目录对账（内置节点/边：冻结核 ≡ 查询期活反射；fields ↔
  `signature_params` 过滤公式显式断言）。
- 同频：AGENTS.md「添加能力」① 签名纪律句入册、红线 2 反射单源句；driver README
  能力注册/签名约定/类型门叙述。

### #140 重审收尾：词表降级收编声明不一致 + 错误候选清单（#140）

- 声明与注册不一致改抛专类 `DeclarationMismatch`（⊂ ValueError，既有捕获姿势
  不破）——`module._ns_info` 词表枚举守卫区收编该类降级：坏声明不再击穿整表
  `list_nodes`（键歧义等其它 ValueError 不受影响，照常 fail-loud）。
- 错误文案补「实际注册键」候选（错误路径注册表身份扫描——红线 1 带候选清单；
  未装饰目标无候选零附加；`fn` 仅错误路径诊断消费，成功路径零消费）。
- 文档同频：discover.py 词表来源句（CR 兜底范围澄清——加载期拦注册表不自洽，
  名段逐字一致仍 CR 面）；data/model pyproject 注册注释 `@handler` 旧词 → `@node`。

### 身份桥退役：解析流注册表单源化（#140，BREAKING·SDK 面）

- **BREAKING**（SDK 面）：`node_of`/`edge_of` 与身份弱键映射 `_NODE_OF`/`_EDGE_OF`
  全删——#126 桥退役，注册表（`NODE_REGISTRY`/`EDGE_REGISTRY`）成为唯一身份面；
  持 fn 取调度面的消费方（含测试桩）迁注册表键直查。
- **BREAKING**（加载期语义，翻 #126「只查装饰不比名」裁定）：discover ③ 首载
  `ep.load()` 后按**同一 node_key 二次查注册表**（与 ① 暖命中同一查表动作——
  发现表键 = 注册表键），load 后 miss = 统一「声明与注册不一致」fail-loud（带
  ep 与键上下文）：未装饰 / 错 namespace / 别名键合并一类，错 namespace 自 CR
  静默面升格为加载期 fail-loud 门；与 #86 无别名政策对齐（别名 ep 用法退役）。
- 连带迁移：`Context` 直注 Node（`fn=` 构造参数改 `node=`——builder 织入，
  运行期结构性零目录查询不变）；conftest `_FnCatalog` 替身缝改注册表身份扫描
  （`.fn is fn`），装饰桩注册键跨模块唯一成为替身缝顺序不变式。

### discover 三步解析流：注册表暖路径 + raw 形参退役（#137）

- `get_node`/`get_edge`/`find_edge` 解析三步化：① `NODE_REGISTRY`/`EDGE_REGISTRY`
  暖命中直返（进程内已加载节点 dict 量级——实测暖查询 0.1µs vs 目录现算
  6.8ms）→ ② miss 才现算目录表（纯元数据、安装即时可见保留）→ ③ 首载
  `ep.load()` 触发装饰注册，身份桥（`node_of`/`edge_of`）校验装饰性，坏声明
  fail-loud 语义不变。热更新赌注收窄一行：卸载/后组覆盖的即时可见让渡
  （暖命中驻留——进程内中途卸载或覆盖包比安装更罕见）。discover 对注册表/
  身份桥改模块级导入（#132 后 node/edge 零 discover 依赖，导入期无环）。
- **BREAKING**（目录访问面）：`raw=True` 双返回面退役——`get_node`/`get_edge`/
  `find_edge`（及 `EntryPointCatalog` 同名方法）只返 `Node`/`Edge`，overload
  双面收单面；原函数消费方迁 `.fn`（生产位：model/registry、builder、module、
  cleaning_session）。

### MLP hidden_dims B006 消解：Sequence 契约 + 不可变默认（#133）

- `mlp.py` hidden_dims：`list[int] = [64, 32]` → `Sequence[int] = (64, 32)`
  （注解写领域契约——只读、list/tuple 双收；默认值写安全形态——不可变防共享；
  与 `_mlp.py` estimator 面既有选型对齐）。per-file-ignores 的 mlp.py B006
  过渡条目删除，`ruff --select B006` 全仓零命中。YAML/list 调用面零影响
  （signature_params 只读 default 值、hidden_dims 不进搜索空间）；get_node_info
  default 展示串 `[64, 32]`→`(64, 32)`（纯展示）。
- 本票①半边（Upstream.expect 只写字段退役）**已由 #131 随票收编落地**
  （PR #135）——票面与雾行以 #131 记录为准。

### 注册表归居契约模块（#132）

- `NODE_REGISTRY` → node.py、`EDGE_REGISTRY` → edge.py——注册状态随契约 + 唯一写者
  （装饰器）同居，与 `_NODE_OF`/`_EDGE_OF` 同所；**装饰期零 import discover**（含传递边——`context.get_node` 降惰性；直接边已删，
  导入期图无环：discover 对 node/edge 仅函数体取桥）。
- discover 纯消费面（entry points 元数据 + 装饰性校验）；钉测：新鲜进程 import
  node/edge 不拉起 discover。

### 编译期 DAG 参数校验：Builder 消费 Node.param_schema（#131，契约扩展）

- `Node` Protocol 增 `validate_params(params, nid)`（编译期校验面）；装饰期派生
  `_params_contract` 一次出两面——校验模型（私有 `_params_model`）+ JSON Schema
  投影（`param_schema` 公开不变）——**一次参数解析、多处复用**（builder 校验 /
  词表投影 / MCP info）。
- Builder 编译门：M1 撞名邻位新增已知键子集校验——坏类型/约束越界在
  `Builder.build()` 即拒（带节点 id + 字段路径），不再等节点执行；
  **语义与运行期防线同源**：lax 强转一致、未知键放过（`**_` 兜底惯例——
  编译期不得比运行期严）；`validate_modeling_steps` 免费获得零执行成本的
  结构化违规反馈。
- **雾三随票收口**：`Upstream.expect` 纯只写字段退役（零读方证据：全仓唯一
  `.expect` 读点属 UpstreamMeta 词表投影）——Upstream 收窄 meta+call+nodes，
  两工厂与 builder/测试构造位同步。

### BREAKING：SDK 契约名对齐——handler.py→node.py、Handler→Node（#128）

- 文件与契约名对齐：`driver/handler.py` → `driver/node.py`，`Handler` → `Node`；
  **装饰器名不变**（`@handler`/`@edge` 是动作，`Node`/`Edge` 是契约）。连带件随改：
  `handler_of`→`node_of`、`HANDLER_REGISTRY`→`NODE_REGISTRY`、`_HandlerEntry`→`_NodeEntry`、
  `_HANDLER_OF`→`_NODE_OF`（99 文件 sweep，三方名如 StreamHandler/HandlerResult 词边界天然免疫）。
- `node.py` 的 `Context` 导入移 TYPE_CHECKING（future-annotations 下纯注解面）——模块级
  依赖链收窄为 `node → binding, config`；防环双向（装饰体↔discover、_checked↔身份桥）
  全在函数体。as-built 导入链叙述落 node.py 模块 docstring。
- **BREAKING 面**：`from mflowy.driver.handler import ...` 的 SDK 导入需改
  `mflowy.driver.node`（仓内零残留；运行语义零变更——纯改名）。

### BREAKING：Handler/edge 契约升级 Protocol + 注册表，装饰器零属性挂载（#126）

- `@handler` 构造 `Handler`（fn / middlewares / param_schema / ns + `run` 调度链）注册进
  `discover.HANDLER_REGISTRY`（`<node_key, Handler>`），`@edge` 同构（`Edge` 单面 fn）——
  **装饰器不再向 fn 挂任何属性，原样返回**；fn 直调语义与直接 import 入口不变。
- **消费面迁移**：`Context.handler` = Handler 实例（`run` 即调度链；fn 种子经身份桥
  解析——运行期零目录查询的结构性质保持）；`get_node(key)` 非 raw 返回 Handler
  （raw 不变返回原函数）；`get_node_info` 的 schema 取 `Handler.param_schema`
  （装饰期自签名派生，派生器自 module 单源迁居 handler）；中间件链类型别名
  `Handler` 让名 **`Chain`**（SDK typing 导入面，builtin/data/model 中间件同步改名）。
- fn↔entry 桥 `handler_of(fn)` / `edge_of(fn)`：**身份弱键映射**（同名不撞、随 fn
  回收）——测试与持 fn 消费方取调度面的唯一姿势。
- 目录加载校验语义等价迁移：旧 `hasattr(fn, valid)` → 身份桥本尊判定——**只查
  装饰性、不比名**（entry point 名与装饰器 namespace 逐字一致仍是 CR 交叉检查面）。
- **BREAKING 面**：读 `fn.handler`/`fn.namespace`/`fn.node_key`/`fn.edge` 的代码
  （仓内零残留）与 `Handler` 别名按链类型消费的 SDK 导入需迁移。

### 估计器 mixin 纯化：剥 TYPE_CHECKING 借基（#122）

- 4 个估计器族文件（`_random_forest`/`_catboost`/`_lightgbm`/`_xgboost`）删除
  「类型面借具体基类」惯用法（`if TYPE_CHECKING: _XBase = 具体类 else: object`）——
  运行期零作用（super() 按实例 MRO 协作解析至组合方基类），纯类型面变更。
- mixin 内 **20 处跨界引用 / 17 条标注**（`super().fit` ×5、组合方属性 ×14、self 作参数 ×1；3 行含双属性引用）
  统一改为**组合方契约局部标注**：带码 ignore + 理由「由子类 (mixin, 具体估计器)
  组合满足——mixin 定义期不可见」。
- 裁定（地图主人重申）：mixin 内部是否满足需要的属性和方法，取决于**组合方**
  而非 mixin 本身——fit 与属性同理；借基是大于局部 ignore 的蒙眼布（让整个三方
  成员面静默可见）。

### BREAKING：运行期值级复核退役——静态安全哲学收口（#57）

- `binding.check_value` / `_allows_none` 删除，`Upstream.node` 去 `check_expect`
  形参、`Upstream.edge` 去边返回值复核、`builder._match_upstream` 收窄二元组
  返回——**此前运行期拦截的「签名撒谎」（声明 X 实产 Y）不再 raise**，错型值
  按鸭子协议流入下游（远处异常或静默语义漂移）。
- 裁定依据（地图主人重申）：**Protocol 代 ABC = 选择静态安全、舍弃运行时保证；
  运行时保证仅业务需要时做**。编译期 `verify_output_hint`（agent 校验反馈面）
  与共用件 `_unwrap` 保留；值级复核的独捕面（第三方插件签名 / Any 洗白 /
  失败定位）无业务需要存活——复活触发器：①第三方插件生态出现 ②首起静默
  语义漂移事故。
- 同频：binding/context 模块 docstring、driver README（execute 行/解析核心/
  血缘三处——顺带修正陈名 `_check_value`）、`test_binding.py` 值级用例组
  （5+1+1=7 用例）随层退役，编译期契约基线不动。

### CI 矩阵：无 torch 环境 job + 覆盖率报告（#43）

- **no-torch job（M6 环境级验收）**：`uv sync --all-groups --extra data --extra model`
  （显式 extras 不带 `[deep]`）后跑两条证据链——① `mflowy list_nodes` 词表降级断言：
  `model.MLP` 条目 `upstream=null` + ns 级 `requires={"MLP": "deep"}` 标注 + RF 条目
  可内省复核（防整体 import 崩溃的假阳性）；② RF 建模会话全链冒烟
  （`test_modeling_session.py`，MLP 用例 `importorskip` 自跳过）。引擎侧降级契约
  #49 已钉测，本 job 补真无 torch 环境证据。
- **覆盖率报告**：dev 组新增 pytest-cov；`make test-cov`（term 日志 + coverage.xml）
  由 CI test job 消费并上传工件——先有数字不设阈值。
- 环境矩阵维持 Python 3.12 单版本（既定支撑面，扩版本另议）。

### 类型门回归清零：dev 0.2.0 收拢后 pyright 126 errors（#107）

- **清零 126→0**（#42 压制纪律：真修为主，带码 ignore 5 处带理由——pandas-stubs ×3、
  mlflow 注解 ×1、负例调用 ×1；零规则关闭/降档）。源码面：`Headers` 协议值类型
  str→object（SDK `_meta` 是 TypedDict，原 `Mapping[str, str]` 注解与「原样透传」
  契约不符）；清洗会话卡形参 dict→Mapping（对齐 `Observation.prior` 类型）；
  `clean_finish` FinishVerdict isinstance 收窄；optuna `best.value` 断言收窄；
  `mlflow.sklearn` 别名导入；`route_task_dir` 双 wrapper 重命名消 redeclaration。
  测试面 100 处：FinishVerdict/dict 分支 isinstance 收窄、Optional 先断言后访问、
  替身缝位 cast（test double 惯用法）。
- **防再犯**：dev 进 CI push 触发面——本地直合 dev 绕过 PR CI 是本次回归的盲区
  （债务直通 release PR #105 才暴露），push 即跑门把暴露点提前到集成时点；
  Makefile 发布前人肉门序列补 `make lint && make typecheck`（#98 发布门五项缺
  typecheck 的教训）。

## [0.2.0] - 2026-11-04

### BREAKING：工件系统收口（#95）——落盘壳全覆盖 + task_dir 双路由锚定 + OOF/study 归位父 run

- **task_dir 双路由锚定（裁定 B）**：锚由入口中间件 `route_task_dir` 决定，工具以
  `@anchor_task_dir("<参数名>")` 声明锚参数（纯元数据——不包函数，签名/schema 零影响）。
  CLI（runner）：`MFLOWY_TASK_DIR` env **钉住**（`pin_task_dir`，推导不覆写）→ 推导 →
  缺省 `cwd`；MCP（server.register）：推导 → 缺省 `~/.mflowy`。七个数据工具已声明锚
  （compute 五工具 `data`、data_profile/eda `file_path`）。`resolve_data_ref` 的锚定
  **副作用退役**（只读解析）；`LocalJobProvider` 的 YAML load 步扫描降级为「未钉住时的
  推导」；`anchor_of_source` 单源上移 utils/path。
- **OOF 观察面工件挂父 run 工件树（裁定 A）**：run_hpo 的 OOF 图集经 `log_figure`、
  `oof_y_pred.parquet` 经 `log_table` 落父 run；session 镜像目录 `runs/<run_id>/` **退役**
  ——账本 `oof.y_pred` 指针从 session 相对路径改为**run 工件名**（寻址 = (run_id, name)，
  `download_run_artifact` 读回），skeleton `oof_plots` 解析为本地绝对路径。`log_table`
  不保索引——OOF parquet 按位对齐（行序 ≡ y_train 行序，与 #66 holdout 副产物同契）。
  `oof_plot_album` 签名去 `out_dir`（挂当前 run）。trial 嵌套 run 工件面零记录为原则。
- **study storage 随父 run 只读投影**：每次 run_hpo 上传 sqlite 副本（`log_file` 落盘壳新
  原语，工件名 = 账本 `study` 字段同名）；session 盘上真源不变（账本 append-only）。
- **figure 落盘纪律单源**：`utils.plot.prepare_figure`（居中微调 + dpi）为 `log_figure`
  （run 工件树）与新增 `save_figure`（会话聚合面直写，Saved 通知走 logger）共用，闭图纪律（#92）随两缝继承；泰勒图
  `dpi=150` 硬编码退役（统一 `DPI`）。
- **metrics 落盘壳**：`utils.mlflow.log_metrics(metrics, *, graceful=False)` 守卫前置
  fail-loud / graceful 直调降级 warning（对齐工件壳语义）；8 处 `mlflow.log_metrics`
  fluent 直调收敛（4 域中间件 graceful、`_log_metrics` 静默跳过→有声降级、HPO objective
  `in_window` 分支走壳）。测试 patch 目标相应移至壳缝 `mflowy.utils.mlflow.log_metrics`。
- session 缓存（split.pkl / loader_*.pkl）转 `write_bytes_atomic`（tmp+rename）。


### 测试内存纪律——图落盘即闭（生产级泄漏）+ 重栈会话 fixture 模块级共享（#92）

- **闭图单源纪律（生产修复，长跑 MCP server 同病）**：三处直存/直 log 图路径从不
  `plt.close`，600dpi Agg 缓冲 ~80MB/张由 pyplot 图管理器进程级持有——`run_hpo`
  OOF 图集（`report.oof_plot_album`）漏 ~5 张/次、holdout 终评（`_evaluation_plots`/
  `_loss_curve` 经 `log_figure`）漏 ~5 张/次、泰勒图 1 张/次。闭图收进
  `utils.mlflow.log_figure` 本体（单源，调用方一次覆盖；log_plot 中间件自身的
  close 变幂等冗余）+ `report.py` 两处直存点；回归钉 = `run_hpo`/`draw_taylor`
  用例断言跑完零开图。
- **取证（#92）**：make test 单进程 RSS 线性爬升至 ~19.8GB 的根因不是（只是）
  fixture 逐用例重建——malloc_trim 一字节还不回、会话末 164 张未闭图实锤活引用；
  闭图后 model 包峰值 13.7GB → 2.1GB。
- **fixture 收敛**：`test_finish_report.py` 终态会话 `delivered`（4 条训练管线）
  function → module 作用域（只读族一次构建；finish 终态化用例走独立
  `delivered_fresh`/`delivered_rf`），scope 回归钉入文件；终训副产物按 run_id
  跨用例读回的前提 = tracking 库模块级共享（`shared_tracking`：fluent 随建随还 +
  cwd=tmp 急切建库锚定工件根 + `_pin_shared_tracking` 复指 env——`run_artifacts
  → setup()` 从 env 解析 tracking，env 不钉即断 fan-in 寻址）。单文件 40 → 11
  条管线，峰值 10.4GB → 3.6GB、142s → 41s，用例零增减、断言零弱化。
- **model 域 tests conftest**：用例间 `gc.collect() + malloc_trim(0)` teardown
  （非 glibc 优雅降级）；状态机型文件（test_modeling_session 34 用例等）结构性
  不可共享会话，靠闭图 + trim 兜底——model 包 449 用例峰值 2.05GB / 7:17。
- **顺手修（conftest 裸名 import 影子）**：`tests/test_logging_setup_isolation.py`
  裸名 `import conftest` 在多树收集下被任一子树同名 conftest 影子（sys.path 按
  收集序插入各测试目录——本 PR 新增 `model/tests/conftest.py` 即触发两钉红），
  改为显式按路径加载根 conftest。
- **验收（#92）**：`make test` 单进程峰值 ~19.8GB → **2.1GB**、全量 ~15min →
  ~9:40，连续全绿（1236 用例）；`test_finish_report.py` 12 → 13 用例（+scope 钉
  +零开图断言）、原断言逐字保留。
### 依赖钉版：mlflow 收窄至 3.15 系（`~=3.15.2`）

- 默认库 `~/.mflowy/mlflow.db` 跨安装形态（dev 锁 / uvx 自由解析）共享单文件，
  mlflow schema 随 minor 前移——开放上界（`>=3.14`）曾致 uvx 解析 3.17 前向迁移
  共享库、dev 全部观测工具 schema 失配不可读（2026-10 事故，库已降级修复）。
  mflowy 分发形态（uvx/Docker/专用 venv）为 app 语义，窄钉零依赖地狱代价。
- 升级纪律：锁（uv.lock）与声明（utils pyproject）**必须同 PR 同步**，并在
  CHANGELOG 记 BREAKING（共享库 schema 迁移影响全体用户）。
- 迁移小项「utils mlflow 转 extra」撤销：tracking 是节点 run 刚需（每个 handler
  开窗），hard dep 合理。


### BREAKING：执行结果目录收敛——DAGResult/NodeResult 最小元信息面 + 工件落盘壳 + mlflow 观测双工具

- `DAGResult.graph` 保持 mermaid 展示形态（机器投影 = `to_graph()` 公共方法，
  `walk()` 单源）；结构图持久化为实验级 tag `mflowy.graph`（rerun 覆写 = 本实验
  最新结构，超长截断——实验 tag 上限 5000 与 run tag 8000 双上限由
  `clip_tag_value` 参数化吸收）。字段 `experiment_name`/`description` 删除——
  入参回显零熵，实验本体经 `get_experiment` 查询。
- `NodeResult` 新增 `artifacts: list[str]`：account 时刻 run 工件树**顶层清单**
  快照（`Tracking` seam 增 `run_artifacts`，FakeTracking 同步记录面）——工件索引
  真源从 output 的 Saved print 行转为结构化字段；run 描述（`mlflow.note.content`）
  ≡ `NodeResult.output`（account 期经 seam `describe_run` 回写，单源合成）。
- 失败 output 语义 **替换→追加**：崩溃前 partial stdout 暂存于 `Context.partial_output`，
  失败记账 = partial + `repr(e)`（崩溃前证据不丢，描述同构）；`mlflow_log`
  中间件的失败 note.content 写入删除（单写者收敛至 account）。
- `utils/mlflow.py` 工件落盘族收编 `_artifact_shell` 模板方法：active run 守卫
  前置 fail-loud（mlflow fluent 无窗口会静默自建孤儿 run 污染默认实验）、目的地
  解析（file://→本地路径；直写形态远程根 fail-loud）、Saved 通知 print→logger
  （stdout 纯业务数据通道）。`log_table/log_dict/log_pickle/log_figure` 改写为
  壳上薄封装（`graceful` 旗标 = 直调形态跳过 + warning），新增 `log_model`
  （恒 graceful，返回 model_uri）；`_artifact_local_path` 删除（无 run 静默回退
  分支消灭）。`_pipeline`/`log_task_evidence` 散守的 `if mlflow.active_run()`
  全部归一为 graceful 调用；终评图/损失曲线另设窗外整跳过（判 fluent 窗口而非
  seam——副产物族计算+落盘整体随 run，直调形态不白算）。
- **Context 身份重设计（BREAKING）**：`Context.id` = `node_key#分组序号`
  （ContextVar 分组计数 map，从 0、构造期定终身——跨配置编辑不漂移；身份与执行地址
  `run_id` 彻底分离）；`node_id` 属性与双语义 `id` property（执行后=run_id）删除，
  图键/失败地址/mermaid id 统一为 `Context.id`，mermaid label 改用
  `Context.label`（`conf.name` 优先、降级 id）；`_node_key()` 辅助函数消亡。
- **mlflow 观测工具 4→2**：`list_runs` 并入 `get_experiment`（实验仪表盘 = 本体 +
  run 状态目录 `node/name/run_id/status/start_time` 五元组投影，对标 NodeResult、
  过滤/翻页随迁；params/metrics 全量返回收窄为 `get_run` 下钻）；`list_run_artifacts`
  并入 `get_run`（详情 = params/metrics/tags/artifacts 产物目录）；`read_artifact`
  退役（工件后端为本地文件树，内容直接访问）。工具总数 33→31（平台 9→7）。


### 类型门落地：pyright basic 全仓清零 + CI/本地/pre-commit 三工位门（#42 决议→#91）

- pyright（dev group 锚版本）+ 根 pyproject `[tool.pyright]` 单源配置：
  `typeCheckingMode="basic"` 钉死（pyright 1.1.373 起裸跑默认已漂移 standard）、
  `venvPath`+`venv` 键锚解释器（`pythonPath` 键静默失效——实测产 975 幻影
  missing-import）；门范围 = packages + tests（scripts/examples 不进门）。
- 存量 604 errors（basic+venv 锚定，dev @ 1306d919）一次性全清：真修为主
  （签名对齐/ParamSpec 装饰器/mlflow 子模块导入/matplotlib 源头导入/Optional 断言/
  Literal 单源化），第三方 stub 缺口与负例测试逐处 `# type: ignore[规则]` + 中文理由；
  config 关规则与目录降档为零。`reportFunctionMemberAccess`（挂属性惯用法，
  standard 档）不进门不修。
- 三工位：CI 独立 `typecheck` job（与 lint/test 平级，error 阻断、warning 不挡，
  release/publish 需求链同步扩）；`make typecheck`；pre-commit 全量 pyright
  （~15s，`files: (\.py$|^pyproject\.toml$)` 过滤——纯文档提交零开销）。

### BREAKING：插件注册静态化——构建期 hatch AST 扫描钩子退役，entry points 手写声明（贡献者工作流 breaking，词表零破坏）

- 两份 `hatch_metadata.py`（105 行 × 2，构建期扫描 `@handler`/`@edge` 生成注册）删除：
  83 条 `mflowy.builtin_plugins`（节点 78 + 边 5）+ 24 条 `mflowy.tools` 全部手写为
  两域 `[project.entry-points.*]` 静态声明；`[tool.mflowy.tools]` 过渡表、
  `dynamic=["entry-points"]`、`[tool.hatch.metadata.hooks.custom]` 接线一并退役。
  内置包与第三方插件的注册方式就此同构（CONTEXT 插件词条宣称成真）。
- 动机（架构复审候选 4）：插件身份协议原存两套实现 × 两份已互相漂移的拷贝
  （model 侧 `_EXCLUDED` 多 `"base"`）、纪律同步——删除构建期实现即删除第二份协议定义。
- 约束机制（决策：不设镜像测试）：注册漂移由 CR + `list-nodes` 交叉检查兜底——
  AGENTS「添加能力」CR 必查三项：新 `@handler`/`@edge` 必有声明、声明必有函数、
  namespace 段与装饰器逐字一致。机器兜底仅加载期单向（声明指向未装饰函数即
  fail-loud，`_load_ep` 属性校验）；忘声明/错 namespace 为静默面，靠 CR。
- StepType 构建期防线随钩子退役（类全仓已不存在，残留旧签名 import 期 NameError）；
  `tests/test_hatch_metadata.py`（钩子契约）随之删除；根 pyproject 两处 E402 豁免清除。
- 迁移验收：`list-nodes` 输出与 107 条 entry points 名单迁移前后逐字节一致（diff=0）。
- 工作流变化：新增/删除/改名节点或边 = 代码 + pyproject 声明同 PR + `uv sync`
  （原为仅 `uv sync`）；运行时行为、词表、YAML 兼容性零变化。

### 工具清单单源——平台工具并入 `mflowy.tools` 组静态声明（注册机制统一，词表零破坏）

- 架构复审候选 2 落地：平台 9 工具从四处分叉推导（server 手写元组 / runner
  `vars(platform)` 反射 / `test_workspace` 反射并集 / `test_tools_schema._TOOL_FNS`
  手列清单）收敛为 mflowy-mcp pyproject 静态声明（与域包、插件注册同构）——
  server 注册塌缩为单一 entry points 循环，runner 分发表同源消费；「忘改 server
  元组 → 工具对 runner/词表可见、对 server 静默缺失」的非对称病灶结构上消除。
- 镜像契约 `packages/mcp/tests/mcp/test_platform_tools_registry.py`（与域侧
  `test_{model,data}_tools_registry.py` 同构）：声明面（安装态 entry points，按
  dist=mflowy-mcp 过滤）↔ 函数面（tools.py 反射）双向差集——写函数忘声明 /
  删改不同步 / 声明后未 `uv sync` 三态皆红。
- 撞名 fail-loud（红线 1 + 单源）：`mflowy.tools` 组重名在唯一消费口
  `_tool_entry_points`（tools.py 私有共享件，server/runner 两入口同源）显式拒绝
  带候选清单——SDK 撞名仅警告且静默保第一个、runner 旧 dict 后写覆盖即静默遮蔽，
  两处旧静默面随触碰收敛。
- runner 分发表顺带清净：旧 `vars()` 反射把 tools.py 顶层导入名（`Annotated`/
  `Field`/`Path`/`asdict`/`sha256_of`）混进分发表（命中在 bind 阶段炸，属泄漏面），
  塌缩后消失。
- 工作流变化：新增平台工具 = mcp/tools.py 公有函数 + mcp pyproject 声明 + `uv sync`
  （并入上一条 BREAKING 的贡献者工作流叙事）；AGENTS 小项「runner 分发表与 server
  注册清单同源」划掉。迁移验收：`tools/list` 名集迁移前后等价（33 工具零丢失——
  旧 server 手写元组本不含导入泄漏名，泄漏面仅存于 runner 反射分发表，随塌缩消失）。

### 取值闭包工厂居 binding——`Upstream.node`/`Upstream.edge` 构造单点，builder 编译面零运行期代码

- 架构复审候选 3 落地：取值闭包（惰性拉起/哨兵去重/值级复核/collect-single 整形）
  原定义于 `builder._fetch`、在 `context._bind` 调用帧执行——内核最差 locality；
  且「`Upstream.nodes` 与闭包捕获同一列表」这条剪枝语义不变式只活在三处注释里，
  全仓零测试保护（test_prune 手搭图的 call 恒为 `lambda: None`，不触该机制）。
- 工厂 `Upstream.node`/`Upstream.edge`（classmethod，与 `meta.type` 的
  "node"/"edge" 值域镜像命名）住 binding——域词汇背书：CONTEXT「拉起」词条
  自定义拉起为**绑定期**行为；binding 模块头本就认领「签名 → 边声明 → 注入值」，
  工厂补全该终点。宪章「零图知识」精确化为「零拓扑知识」——闭包按鸭子协议消费
  上游执行面 `result`/`run`，零 Context import、零目录查询。
- 列表同源从注释约定变为工厂的结构保证：`nodes = chosen` 全仓唯一焊点，剪枝
  就地过滤（`Context.rewire` 的 `nodes[:] = ...`）两边天然同步——同源焊点唯一，
  直构数据类不产生取值闭包。`Upstream` 保持普通 dataclass——测试手搭缝位
  不变（红线 7），工厂为生产代码唯一构造点。
- `_NOT_RUN` 执行哨兵随迁 binding 与 `MISSING` 并肩（两大协议哨兵同居单源）；
  builder 两处构造点各缩一行、删 `_fetch`/`edge_call` 定义，从此不含运行期执行的
  代码。行为零变化（纯搬移 + 封装）。
- 测试：行为级钉测入 test_prune（真实 Builder 图 + disabled 剪枝 + run：收集边
  只剩幸存成员、可选成员全灭落签名默认值、被剪节点零拉起）；工厂直测入
  test_binding（拉起恰一次/同源过滤即视/MISSING/collect-single/撒谎复核点名）；
  test_compile 八处私有 `_prevs`/`_nexts` 断言公共化（`prevs`/`nexts` 只读视图）。

### Added（#88）：login 平台工具——OAuth 2.1 身份地基（PKCE S256 + loopback + token 库）

- **`mflowy.utils.auth`（纯 stdlib）**：平台 OAuth 2.1 客户端——授权码流
  （PKCE S256 + 本机浏览器 + `http://127.0.0.1:{随机端口}/callback` loopback 回调 +
  state 一次性比对），token 端点 JSON 体（契约明确非 RFC 6749 form）；
  凭据 `~/.mflowy/credentials.json`（0600）单文件 tmp+rename 原子整替（umask 收紧，
  token 不经 644 窗口）。refresh 轮换制：刷新/落盘临界区持文件锁 + 双检——并发二次提交
  旧 refresh 会触发平台复用检测（账号全端下线），同机多 agent host 并发是现实场景；
  `invalid_grant` 即清本地凭据并以 `LoginRequiredError` 引导重登（文案含全端下线风险）。
- **`login` 平台工具**（MCP/CLI 同面，`mflowy login '{}'` 免注册即用）：幂等——同环境已有
  凭据直返 `already_logged_in` 不弹浏览器，`force=true` 强制重授权；scope 缺省拉服务端
  元数据 `scopes_supported` 全目录整体同意（客户端不拷目录，单源）；`base_url` 三级优先
  参数 > `MFLOWY_BASE_URL` > `http://localhost:30080`。JWT 本地仅解码展示不验签
  （资源侧统一验）。平台工具 8→9、总数 32→33。
- **消费接线点**：`ensure_access_token()` / `LoginRequiredError` 供未来远程 JobProvider
  附 Bearer（401 单次刷新重试），不重造轮子。
- 顺手收敛：mcp 包显式声明 `mflowy-utils` 依赖（tools.py 直接 import 的存量缺口，红线 6）。
- 复审收敛（同批）：凭据损坏面 fail-loud——未知字段（篡改/损坏信号）与非法 JSON 按类型
  二分（`CorruptCredentialsError`，删文件重登即恢复；login 撞损坏文件自动重授权覆写）；
  `client_id` 入库（凭据自包含）；token 端点响应 → `Credentials` 构造单源（登录/轮换共用，
  claims 缺失按路径回退）；刷新锁等待上限 10s → 45s（须大于单次 HTTP 30s——持锁者的合法
  刷新不该让等锁者伪失败）；scope 显式为空（`[]`/`""`）拒绝（静默回退全集 = 静默改写语义，
  收窄请至少给一项）；切换 `base_url` 平台环境落 warning（旧环境凭据被覆写）。

### Fixed（#77）：inverse_optimization 模板补渲染 n_trials/random_seed——调用方传参不再被 Jinja 静默丢弃

- LocalJobProvider 恒把 `n_trials`/`random_seed` 放进模板 env，但
  `inverse_optimization.yaml.j2` 全文不引用——Jinja 静默忽略多余 env 键，
  `search_input` handler 默认值（n_trials=10000）恒生效，调 `n_trials=200` 实跑
  10000（红线 1「静默丢弃即缺陷」）。模板 `search_input.params` 补渲染两键；
  StrictUndefined 下 env 缺键即 fail-loud。`test_ns_return_contract.py` 编译测试
  env 补两键，并断言渲染值真进了节点 params（取非默认值 200/7 钉住渲染行为，
  恰等于默认值的断言钉不住）。
### BREAKING（model 域出题面收敛——设计稿 [2026-10-28](docs/research/2026-10-28-model-proposing-convergence-design.md)：问题体征分类学 + 规则表定义权归 propose）

三段落地（model 域出题面对齐 data 域「问题分类学 → 问题规则 → 处方菜单」纪律）：

- **证据先行（段1）**：新增 `theta_probe`——θ* 单次代表拟合（train 内切 val +
  eval，早停剥离防截断上翘段）双产物：train-fit/val **同尺得分对**（同一次拟合的
  苹果对苹果，欠拟合/过拟合 gap 两体征的证据）+ 逐迭代损失曲线；**objective 纯度
  裁定**：诊断只对 θ* 做，trial = 纯训练+指标（user_attrs 只携带搜索证据 scores，
  钉测约束）；diagnose
  四体征算子——`underfit_high_close`（train/验证双高且接近，skill vs 平凡基线
  = 中位数/多数类经 `evaluate` 单源）、`overfit_train_val_gap`（验证显著劣于训练）、
  `underfit_loss_plateau`（曲线缓降/高位——改进亏缺代理）、`overfit_val_loss_rise`
  （先降后升）。族异构降级：OOB 族（RF）无曲线不产曲线体征。**顺手修两处存量缺陷**：
  diagnose 任务分发按裸串 `"regression"` 比较而会话传值形式 `"regressor"`——回归
  会话从未真正走 error_concentrated 分支（现 `parse_enum` 双形式归一）；LGBM 分类面
  `evals_result_` 键序与回归面不一致（`training` 键缺席）——loss curve 提取从未成功。
- **分类学扩容（段2）**：kind 表单源 `propose_rules._KINDS`（kind → 规则体 × 记分牌
  权重）——九问题 kind（八体征/代理 + `error_concentrated`）；`overfit_gap` **改名 `drift`**（BREAKING：
  inner−holdout 差距语义收窄为 CV 乐观偏差/漂移——train/val 直接过拟合证据由
  `overfit_train_val_gap` 承担）；`complexity_edge` 降权 1.0→0.8（参数位置代理，
  物理体征到位）；severity 全域 [0,1]（跨体征记分牌可比）。
- **规则翻转（段3）**：**BREAKING**——基座谓词机层（手拼 `base.*` 伪键）清零，
  候选/首搜/终训/配置编辑/FE 换档全部规则化（`CandidateRule`/`FirstSearchRule`/
  `FinalTrainRule`/`ConfigEditRule`/`FEBaseRule`，salience=-1 垫底——poll 单源记分牌
  排序，两层拼接退役）；问题规则出**处方菜单**（欠拟合 → 换 FE 加信号 + 正则放宽
  下半区；过拟合族 → 正则收窄上半区/pin——`_regularize_rule` 镜像）；**BREAKING**——
  `menu_ctx` 退役：session 只投影状态（findings/光标/combo_key/models 摘要/has_cat）
  进 state_digest，词表（families/fe_vocab/签名/正则面）由规则在 propose 内以环境
  常量现查——`observation.json` 不再持久化词表快照；**BREAKING**——族声明尾巴三并一：
  `.mdp_family` 成为节点函数唯一尾巴（structure 直书内嵌、`regularizers` 并入——
  可选键，无正则旋钮的族可缺省；裸 `.structure`/`.regularizers` 属性退役，#58
  准入键集相应扩展）；`propose_actions` 退成「状态投影 + fire + poll + 合并」。
  MCP 工具面零改动（菜单仍是 Activation 列表）；存量会话 rules.pkl 持旧 kind 集
  （`overfit_gap`）——跨版本升级旧会话 propose 会撞未知 kind fail-loud（unreleased
  内变更，读侧不迁移）。
- 学习曲线居住面（体征 ③⑥）：**暂缓**（净新增计算——train-size sweep 多次拟合；
  待 HPO 预算结构定型后随 diagnose 工具面裁定，设计稿 §3.1）。

### BREAKING（#89 会话基座深化：POMDPSession 统一机制 + 契约 pydantic 化 + 规则引擎归位）

四片落地（M3+M4+M5 迁移行随本条勾销；规格见 issue #89）：

- **壳与契约（①片）**：新增 `driver/session.py` `POMDPSession` 会话壳——六个操作上下文
  （`start_ctx`/`propose_ctx`/`apply_ctx(action)`/`rollback_ctx`/`finish_ctx` + `abandon`）
  承载全部业务流程骨架：终态/预算门（步数口径 `counts_step` 域声明）、prior 冻结牙齿、
  append-only 决策日志（`decision.log` JSONL，追加即 commit 点——接口无删除面）、统一
  盘上写序（工件→日志→观测投影）、`open` 撕裂检测（recent_decisions 尾 ≡ 日志尾）。
  `pomdp.py` 契约层全 pydantic 化：`Observation` 七键封闭模型（`extra="forbid"`——
  **BREAKING**：响应键大改，`should_stop`/`budget`/`allowed`/`card`/`report` 等旧键退役，
  域尾巴归位 `prior`/`state_digest`）、`FinishVerdict`（拒收必附 reason 模型级硬规则）、
  `Activation` 拆 action/params 两成员（**BREAKING**：旧 `proposal` 嵌套键拒收）、
  `POMDPSessionHandle` 改名（旧 `MDPSessionHandle` 退役）。`Candidate` 与 `verify_*`
  手写校验器退役（契约即校验）。**BREAKING**：会话盘上 manifest.json 溶解为
  observation.json + decision.log + rules.pkl（+域状态文件，model 的 board.json）——
  存量会话不迁移，读侧 fail-loud。
- **两域上壳（①片续）**：clean/model 会话模块退成 Handle 薄门面（只填转移语义）；
  版本树 append-only（**BREAKING**：clean rollback 从截断弃分支改为记录+指针回拨——
  版本号日志派生永不复用，弃分支盘上可回）；计划导出从两份手拼 emitter 收敛为
  `Conf → flow_to_yaml` 单源序列化（**BREAKING**：plan YAML 文本形态变为真 YAML，
  Builder 兼容）；finish 幂等重入由壳构造保证（只读重入、双 commit 不可能）。
- **执行面（②片）**：clean apply 从 raw 直调切**单步流编译执行**（Conf→Builder→
  DAG.run——编译期校验/mlflow 观测/捕获/血缘 tag 免费，执行合同 yaml 落会话目录，
  决策记录带 run_id；apply 与导出计划同构）。model `run_hpo` 保持域内 `_pipeline`
  （委派 JobProvider 会 fork 增量 study 真源——约束见 issue #89 评论；终训自 #66 已走
  provider）。
- **规则引擎归位（③片，含余量落定）**：`fire(*keys, o)` 触发上下文 = 观测（两域
  均已落地）；data 域检测层证据富化——problems 问题投影 / columns 列画像（dtype
  族 + skew/四分位 + 方差，计算单源 univariate_facts）/ current_file 写进
  state_digest；**BREAKING**：clean `.repair` 声明面退役——问题知识（问题 → 候选
  动作×参数×档位理由）随问题规则居住（`repair_rules.ProblemRule`，key = kind），
  节点回归纯变换，第三方迁移面 = 注册自己的 ProblemRule；菜单 params 自含
  `load` 引用（当前版本工件），apply 侧新鲜度门（过期菜单 fail-loud 指路重新
  propose），计划导出剥 load 寻址键；`compose_should_stop(when)` 停止规则合成器
  落地两域（data = 缺陷清零、model = 预算耗尽 → action="finish" 提议，salience=0
  垫底）；`rule_engine_rwa` 门面随壳迁移退役（语义住 `POMDPSession.propose_ctx`）。
- **复审收敛（#89 code-review 轮）**：model finish 幂等重入改走壳 finish_ctx 只读路径
  （域侧预检删除——构造保证落地为事实）；无冻结 combo 的 finish 从 raise 改**拒收
  verdict**（与 data 域 θ 门同一契约路——拒收后可续局）；data plan.yaml/steps 合同
  原子写；`SessionStore` 裁到 GC 面（manifest 读写面随溶解退役）；决策记录形状单源
  `_record`；两域函数组 ≡ Handle 协议的结构等价钉测入 `test_workspace`；命名消歧
  （model `_state_projection`/`to_activation` vs data `_refresh_state_digest`/`menu_of`）。
- **JobProvider 协议扩展（#89 ①修正裁定落地）**：`hyperparameters_search` 新增
  `study_storage_path` 可选参——非空时经 ContextVar 锚定，`study.search()` 建 study
  走该 sqlite + `load_if_exists`（一库一 study，名 = 路径词干）：**同一路径重复
  搜索增量续跑，复用前史**（对同一模型多次 HPO 不从头采样）；贯通
  Protocol/LocalJobProvider/MCP 工具面（缺省 None 行为不变——内存态一次性）。
  会话 `run_hpo` 维持域内直连（其增量语义已由 `_open_study` 承载）；本扩展为
  provider/tool 路径补齐同等能力，远程实现者应将参数转发到 runner 侧等价锚点。
- **M5 版本树语义（④片）**：clean 单记录版本树（apply 记录含 v/file/sha/run_id）；
  model `save_loader`/`plans/final_*` 原子写 + **带冻结世代文件名**（
  `loader_{key}_v{n}.pkl` / `final_{key}_v{n}.yaml`——freeze-once ⇒ 恒 v1，世代位
  显式留名与 data 域 v{n}.parquet 同纪律；verdict/loader 指针引用带世代实物名）；
  rollback-after-rerun 一致性用例两域钉测。
- 工具层与 MCP schema **零改动**（六操作门面签名不变——包装层 model_dump 适配）；
  同频：CONTEXT 词条（会话基座/观测摘要/决策日志/Observation·Activation/规则通俗
  定义）、AGENTS（分层表/迁移表）、driver·data README、clean skill、防回潮豁免注记。


### Changed（剧本层调用治理）：六域 skill 置 `disable-model-invocation`

- clean / model / model-brief / research-review / survey / kaggle-competition 六份
  剧本 skill 的 frontmatter 置 `disable-model-invocation: true`——回合循环与问卷/
  评审剧本改为**用户显式触发**（点名 skill 或斜杠形式），agent 不再按 description
  自动装载（剧本入口是用户意图，不是模型猜测）；M11 剧本引用面测试不受影响
  （frontmatter 不属引用面）

### Changed（术语统一 MDP→POMDP）：两域会话 prose 全量改称 POMDP

- 术语权威 CONTEXT.md 会话词条改称 **POMDP** 并增可观测性界定：引擎簿记态（光标/
  账本/预算）对 π 全可见，环境真态部分可见——可观测性以域观测函数覆盖面为界
  （清洗 = 检测算子族、建模 = diagnose/规则面；态混叠），终局 U 依赖观测未含的态
  （「干净但不可学」即其症状）；MDP 入词条 _Avoid_（旧称）
- 全库 prose 清扫：AGENTS / 根 README / 两域 README（新增「会话 POMDP 元素表
  （O/S/A/P·T/R）」——各域五元组定义与居所锚点）/ driver README / skills ×5 /
  docs（research-flow / roadmap）/ 代码 docstring 与模板注释 / pyproject 包描述；
  活性文档零 "MDP" 残留由 `test_workspace` 断言钉住（防回潮）
- **代码锚点**：`mdp_session` 模块随术语统一改名 `pomdp.py`（import 面 `mflowy.driver.pomdp`；
  rules.pkl 等盘上件引用 `rule.py` 与域规则模块的类路径，均未移动——存量会话不受影响）；`MDPSessionHandle`、
  `.mdp_family` 声明键等标识符保持原名（SDK 面最小破坏）；kaggle 外环 skill 的「外环 MDP」为该 skill
  自身的形式化设计词，不在清扫面
- 本 unreleased 区历史条目措辞随之统一（已发布版本不动——追加式纪律）

### Changed（#58）：model 会话引擎收敛——headers 契约单源自省 + 族目录准入校验 + 陈账清理

- **JobProvider headers 契约清单从 Protocol 自省单源生成**：工具×方法清单、样本
  参数、录制替身全部由 `JobProvider` 协议运行期自省派生（`test_tools_headers.py`
  手抄两份清单退役）——新增 compute 方法契约测试自动覆盖；新增工具包装签名 ⊇
  协议参数的一致性断言；`Headers` 类型别名（protocol/local 两处签名共享）
- **族目录准入校验**（K8s CRD 风格）：`registry.families()` 聚合期对 `.mdp_family`
  声明做形状校验——structure 必含 `cat_support`（值域 native/encode）/
  `scaling_need`(bool)/`min_samples`(int)，`machinery` 必须 callable，可选
  `fit_params` = fit kwargs 工厂（callable）；未知键拒绝；坏声明 raise 附声明
  义务指引（错误信息与 model README「族目录声明义务」节同源）。会话中途裸
  KeyError 的故障面消失；`session._machinery_of` 的运行期补查退役
- **build_ct 机械单源**（US11）：FE 槽位→节点装配改经运行期词表现查
  （`FE_CAT/NUM_NODES` 唯一真源——词表/DAG/装配同一解析面），并修复 target
  编码档在会话装配路径的存量断裂（`target_encoder` 的必需参数从 spec 的 y 面
  注入——会话恒单目标）；registry 陈账清理（families() docstring 已退役资源轴
  描述删除）

### BREAKING（#65）：清洗 POMDP 度量层修复——Φ 单调化 + 先验观测 + dry_run 退役

- **Φ 聚合按 kind 均值改求和**：消除任何缺陷都使质量分不降（单调奖励）——三反例
  （删常量列/修共线弱对/单修低缺失列）钉成回归测试；数值整体下移、定义域放开为
  (−∞, 1]，θ 门跨数据集按相对差分读数
- **新 detect 词表节点 `range_violation`**（声明式轴）：问卷 `expected_ranges` 的
  物理违例进观测通道（severity = 列越界率；会话路径由引擎从冻结卡注入，DAG 路径
  显式传参）；kind 进 KINDS 闭合注册表与记分牌默认权重（0.10）；
  `range_clip`（温和档）/`drop_outside`（激进档）挂 `.repair` 进候选菜单——
  判断与执行同尺，问卷先验全程可观测
- **`clean_apply` 的 dry_run 参数退役**（引擎/MCP 工具 schema/剧本/测试四处）：
  样本预览双合同皆死（安全承诺假 + 忠实性假），失败 apply 本就免费（异常在 commit
  点前抛出——不耗预算、不落版本、不进决策日志）；试探走「提交 → 免费 s′ →
  负 reward 即 rollback」
- **propose 证据富化**：`.repair` 生成器产出支持可选三元组
  `(params, label, extra_evidence)`；`fill_median`/`fill_mean` 候选携带目标列
  偏度/四分位（计算单源 = statistic 域 `univariate_facts`，候选侧只投影）——
  稳健档 vs 正态档的选择从标签变读数
- 会话 finish 的终态质量分与回路同尺（卡权重 `defect_priority` 全程生效）
- 剧本引用面谓词入 AGENTS.md：操作 skill 参考节只允许工具名与入仓文档
  （clean skill 的实现路径引用已删除）；两份被剧本引用的 POMDP 设计快照
  （`docs/research/2026-10-09-{cleaning,modeling}-mdp-design.md`）随 .gitignore
  反向规则入仓（M11 引用文档入仓），`test_workspace` 增文档引用面断链测试

### BREAKING：规则引擎重构——散落规则聚合到实例（Rule/Activation/RuleEngine + load/save 纯函数落盘），两域 POMDP 统一 fire/poll 入口

- **实例门面取代节点属性挂载**：规则 = 实现 `driver.rule.Rule` 协议的对象（`key` 身份 +
  `propose(R) -> list[Activation]`），经 `RuleEngine.register(*rules)` 聚合；`fire(*keys, R)`
  按 key 激活（空 = 全部，未知 key fail-loud 带候选清单）产出 `Activation`（proposal 恰含
  action/params + salience 记分牌 + rationale/evidence 解释面）在实例内累积，
  `poll_proposals()` 按 salience 降序（同分 rule_key/action/params 字典序）聚合排空——
  连续 fire 一次取走，菜单确定性不变
- **落盘与状态解耦**：引擎纯内存（注册面 + 激活缓冲，零路径状态）；盘 IO 纯函数对
  `load_rule_engine`（缺文件 → None；损坏/异类 → raise 带路径——自定义规则不允许被
  静默换掉）+ `save_rule_engine`（dumps 后走 `mflowy.utils.file.write_bytes_atomic`
  单源原语）；`rule_engine_rwa(path, default)` 四行门面（整读→用→整写）：缺 pkl 时
  默认实例直接作工作实例（无克隆——调用方传干净模板，域侧每次现建），with 正常退出
  整体落盘（含运行期注册的自定义规则与未 poll 激活态）、异常不落盘。规则须可
  pickle——域规则 = 模块级 frozen dataclass 只持 str/num，词表/注册表对象在 propose 内现查
- **data 域统一接入**（此前 `.repair` 直扫绕开引擎）：`repair_rules.py` 新模块——词表
  `.repair` 声明（知识仍随词汇主人居住）包装为 `RepairRule`（key = `node.kind`）注册进
  cleanRuleEngine 默认实例；`propose_actions` 走 fire（R = defects/card/df）/poll，菜单响应
  形状与记分语义不变（kind_weight×severity 同尺映 salience），会话目录落 `rules.pkl`
- **model 域迁移**：`propose_rules.py` 闭包工厂（`family_rules`/`fe_rule`/`attach_rules`）改写为
  `FamilyRule`/`FERule` 规则实例（key = `family.kind` / `fe.node`），资格门内联 propose 早退
  （skipped 面随内联退役）；`session.propose_actions` findings 驱动层同走 fire/poll，`_rule`
  溯源改 `{rule_key, kind, salience}`（`_score` 分解随 overrides 退役——唯一调用方从未传）
- **BREAKING**：`driver.rule` 旧 API（`@rule` 装饰器 / `RuleSpec` / `Finding` / `fire` /
  `prioritize` / `FireResult`）删除——唯一仓内消费者 model 域已迁移；第三方规则作者改实现
  `Rule` 协议并注册 `RuleEngine` 实例

### Added（utils 原子写原语：tmp+rename 三处分叉收敛为单源）

- **`mflowy.utils.file.write_bytes_atomic` / `write_text_atomic`**：tmp+rename 原子写
  下沉 utils——bytes 基元（tmp 是目标同目录点前缀兄弟，同文件系统 rename 才原子；写
  tmp 失败目标分毫未动，丢弃即回滚）+ 文本门面（先编码后落盘：编码失败发生在任何文件
  效果之前，不留空 tmp 垃圾）；契约测试钉三不变式（成功无垃圾残留 / 撕裂写目标完好且
  垃圾点前缀 / 编码失败零文件副作用）
- **三处分叉收敛**（单源红线："第二份拷贝出现即收敛"）：driver
  `SessionStore.write_manifest` 内联实现（tmp 名 `manifest.json.tmp` 可见款）与 model
  会话 `_write_atomic`（点前缀款）同纪律两写法已分叉——统一走原语，`_write_atomic`
  删除；交付物拷贝（session.py copy-from-file 流式）非 write-from-buffer 算子，单站点
  保持原地
- 行为可见变化：driver 侧 tmp 垃圾名 `manifest.json.tmp` → `.manifest.json.tmp`（点
  前缀隐藏不进 glob/目录列举；仓内无消费方按旧名寻址，非 breaking）

### 单源红线收敛（#83，#66 验收跟进）：cv/max_steps 缺省单点 + 泰勒图名单源 + 片段门共享谓词

- **cv/max_steps 缺省仅由 `CVCard.build` 签名持有**（BREAKING：`CVCard` 卡字段
  `cv`/`max_steps` 转必填——直接构造卡的字段缺省拷贝删除，仓内构造唯一路径本就是
  `build`）：会话开局装配纯透传（未指定即不传——落签名缺省），manifest 的
  `max_steps` 直抄冻结协议值——`int(max_steps or 协议值)` 二次合并退役（显式传 0
  原路径已被 build 校验拦下，现路径直连且钉测试：0 值 fail-loud 绝不 falsy 回退）；
  split 物化折数同步取冻结协议值，不再二落 `cv.split` 签名缺省（两活缺省源 = 漂移面）
- **泰勒图产物名单源**：`report._TAYLOR_FIGURE` → 公开常量 `TAYLOR_FIGURE`，
  `report_skeleton` 图片清单与 `taylor_for_runs` 写入方共用——`taylor_diagram.png`
  字面量全仓唯一（常量定义处，源扫描钉测试）
- **步骤片段门共享谓词**：`templates.parse_steps_fragment` 单源谓词，MCP 预检面
  （`validate_modeling_steps`，结构化 `valid=False` 返回）与提交侧
  （`LocalJobProvider._require_steps`，raise fail-loud）同拒绝集（等价性钉测试）；
  两侧对外行为不变，文案统一收敛到谓词

### M6 缺依赖优雅降级（#49）：families() 装载守卫——无 torch 环境开得了建模会话

- **`registry.families()` ep.load() 增 ImportError 守卫**（同 `module.py` 富化降级
  姿势）：缺 extra 的族条目跳过 + `logger.warning`（消息带条目路径与缺失模块名），其余族照常
  ——修复 CPU-only 安装（无 `[deep]`）连 RF 会话都开不了（`start_session`/`apply_action`
  /`propose_actions` 主链全经 families()）。词表 miss 仍 fail-loud（CatalogMiss 不受影响）
- **`test_registry` 期望集随 torch 可用性二分**：有 `[deep]` 五族全集 / 无则减 MLP；
  新增确定性守卫钉测（monkeypatch 坏 ep → 跳过 + warning，其余族在册）
- 迁移表 M6 行同步（存量违规点已收敛；CI no-torch job = #43 后续交付）

### driver 内核卫生（#55）：撞名 warning 去重 + 内核 print 通道收尾

- **撞名/坏名 warning 进程内去重**（``discover._warn_once``）：目录零缓存现算——
  同键冲突每次扫描都撞真撞名即刷屏；同进程同类同键恰发一次（热更新赌注保持：
  新 entry point 即时可见，去重只针对已报过的同一对）。覆盖三发射点：坏名
  跳过、同组覆盖冲突、跨层撞名（边名撞节点/组）
- **内核 print 通道收尾（通道纪律最后一角）**：``mlflow_log`` 中间件 handler 间
  空行 ``print()`` → ``logger.debug``（非 CLI 宿主不可强制写 stdout）；
  ``DAG.run`` 增 ``echo: bool = True`` 逃生门——节点输出回显属入口层，默认开
  （CLI 观感不变），MCP 进程内调用传 False 保 stdout 纯净
- 钉测：同组覆盖/跨层撞名 warning 两扫描恰一次 ×2；``echo=False`` 零 stdout
  （默认开保持回显）

### lightgbm eval_set → eval_X/eval_y 弃用迁移（#61）

- **统一 seam 迁移**：``fit_with_eval`` 签名 ``eval_set`` 元组 → ``eval_X``/``eval_y``
  双参（keyword-only）；四族 estimator mixin fit 同步（XGB/LGBM/CAT/MLP）——各族
  **原生 API 翻译住在各自 mixin**（xgboost/catboost 原生仍是 eval_set 无弃用，
  lightgbm 走元组形式 eval_X/eval_y 含 train 面，零 LGBMDeprecationWarning）
- ``_inner_cv_plan`` 返回值 (cv, X_fit, y_fit, eval_set) → (…, X_val, y_val)；
  training/oof_predictions 调用点随迁；早停/loss-curve/train-in-eval 行为不变
- 文档措辞同步（lightgbm/random_forest/_loss_curve/_mlp 警示文案——「eval_set」
  →「独立 val」）；钉测：新 seam 下早停行为不变 + catch_warnings(error) 断言零
  弃用告警 + 旧 kwarg TypeError fail-loud（不静默吞）

### run_hpo 盘上 study sampler 随 seed 建（#84 flake 根治）

- **根因**：会话路径 `_open_study` 以 optuna 缺省 sampler（未钉种子的 TPESampler）
  开/载盘上 study，`hyperparameters_search` 的 `seed` 形参在该路径被静默丢弃
  （仅 study=None 的 DAG 路径经 search_ml 吃到）——θ* 每进程吃熵漂移，OOF/findings
  出题面抖动致 `test_findings_drive_only_same_combo` 非确定性失败（跨进程复现率
  ~1/8，与 PYTHONHASHSEED 无因果——hash 序只是碰巧放大了观察差异）
- **修复**：`_open_study` 增 seed 形参，create/load study 均挂 `TPESampler(seed=seed)`
  ——sampler 种子与 objective_id（已含 seed）的目标同一性同步（换 seed = 换目标 =
  study 重开，语义自洽）
- **钉测**：同数据两个全新会话首 trial θ* 逐位一致
  （`test_run_hpo_theta_deterministic_across_studies`——同进程内两个未钉 sampler
  必不同，等价断言真咬合缺陷）；原 flaky 用例补 #84 根因注记
### shap 副产物契约单源化（#85，#67 验收残留）：写读产三方共享件 + 注记缺失 fail-loud

- **契约住 ``shap/contract.py``（单源）**：列前缀（``shap__``/``value__``/``cat__``）
  、记账列（``class``/``sample_idx``/``y_name``）、注记三元组 accessor/组装
  （``annotation_triple``/``attach_annotations``）、3D 分块迭代
  （``iter_output_blocks``——自 plot/utils 迁入，写方分块与 plot 侧迭代共用）、
  宽表组装（``explanation_to_frame``）
- **三方收敛**：写方 ``log_shap_explanation`` 中间件、读方 ``shap__load``、生产方
  ``shap__explanation``/``shap__load`` 全部 import 契约件——前缀/列名/注记字面量
  副本删除（源扫描钉测：三前缀字面量全仓唯一）
- **注记缺失 fail-loud**：裸 ``shap.Explanation`` 缺 ``y_name``/
  ``categorical_features``/``classes`` 时 ``annotation_triple`` 带产出面指引报错
  ——``'-'`` 占位静默回退退役（占位符落盘即断链）
- **顺手卫生**：plot 三消费方（summary/dependence/sample_waterfall）对 ``shap``
  参数遮蔽统一为 ``import shap as shap_lib`` 一式；``search_input`` 死别名
  ``model_loader = model`` 删除
- **round-trip 真跑钉测**：middleware 落 parquet → ``shap__load`` 重建与真算
  逐位同形（2D/3D）

### CatalogMiss 文案带改名去向（#86，#67 验收残留 Story 9）

- 词表 miss 报错文案增**词干后缀建议**（``discover._rename_hint``）：历史寻址键
  （``plot__shap_explanation``/``shap_explanation``/``model__predict``/
  ``model__search_input``）与拼错键按规范化（小写去 ``_``）整等/后缀匹配现行键，
  **唯一命中才出**（多义/无线索不出——宁缺毋滥）；边键以裸名显示。旧键仍拒绝
  （不引入静默别名），用户按报错改一行 YAML 即命中现行键
- 一般情形不退化：纯拼错/不存在键无候选行，``Available`` 清单语义保持；
  happy path 零开销（仅 miss 错误路径计算）；driver-only 环境真目录用例带跳过守卫
- 钉测：假词表机制用例 ×5 + 真目录四键 seam 参数化 + Builder 编译缝文案用例

### 词表身份钉测试补判据③（#87，#67 验收残留）

- ns 返回契约套件（`test_ns_return_contract.py`）新增判据③最小流：
  `load__file → x_y → model__loader → predict → shap__explanation → plot__shap_summary`
  编译通过即断言（单值 ``model`` 边未吃到 predict 的 DataFrame 返回）——predict
  若回流 model ns，类型门经此用例报警而非等用户流炸；测试文件头部判据清单
  （①②③）同步补全

### 基准建模就绪（#70 决策图执行）：B1/B2/B3 三 P0 修复 + 基准任务落仓 + e2e-bench 冒烟门

- **B1（#71）`model__MLP` 全路径修复**：`MLPEstimator.__init__` 增 `random_state` 形参
  （`RANDOM_STATE` 缺省），seeding 住 estimator fit 首行 `pl.seed_everything`——
  handler 级 seed 调用退役（单源，machinery seed_key 契约五族全守：estimator 参数面收
  seed；DAG/会话两路径同语义）。修复前 handler 显式 `seed=` 注入即 TypeError，任何含
  `model__MLP` 的 DAG 必炸（回归于 78d4a089）；验收 = DAG 级执行用例 + machinery 套接钉
  （torch 环境守卫）
- **B2（#72）分类会话 `run_hpo` 修复——y 双态归一层**：sklearn `cross_val_predict` 对
  分类 + proba 方法走 `encode=True`（asarray + LabelEncoder），estimator fit 收标签编码
  ndarray 而非 DataFrame——五族 fit 原假设 DataFrame 全部必炸（RF `.to_numpy` /
  XGB·LGBM·CAT 经 `SubTask.from_y` 的 `y.iloc` / MLP `y.columns`）。修复双层：
  `model/utils.as_y_frame` 共享双态归一（五族 fit 首行）+ `oof_predictions` 单目标压
  1D（sklearn proba 组装对 2D 单列 y 误判 binary label indicator）；MLP 补 sklearn
  分类器契约 `classes_`。验收 = 分类会话五族 run_hpo 用例（此前会话测试全回归任务，
  漏网根因）
- **B3（#73）示例/e2e 漂移收口 + 防再漂机制**：10 个示例 YAML 修漂（`split`→`cv__split`、
  `targets`→`target`、MLP 参数随 #62 退役面更新、house_prices 退役节点换现词表）；
  `examples/multilabel/` 删除（多标签工作负载随 #66 单目标收窄整体退役，保留即永久
  漂移）；Makefile e2e 收口（`modeling_steps` 明文 + `cv` namespace 查询 + 下游目标
  XGB→MLP——树族节点是搜索行为无 final 模型，链内唯一可解释/可预测目标是 MLP）；
  **防再漂**：`tests/test_examples_yaml.py` 遍历示例全过 `validate_modeling_steps`
  （编译即校验钉示例契约）；顺手收敛 P1-6：`load__python` 未知参数 fail-loud（原
  `**kwargs` 静默吞）
- **M10 锚定修复（P1-4）**：`_anchor_task_dir` 增认 load 族 `source` 参数（csv/excel/
  parquet/python 正名，`.py:func` 剥 func 锚脚本目录，URL 不锚）——修复前 `load__python`
  结构性无锚（实验创建即「请使用绝对路径」）；`make e2e` 全链追踪隔离 + 生成器 staging
  出仓（零仓库运行产物）
- **顺手收敛 P1-5**：CAT fit 对 `cat_features` 列 NaN fail-loud 指路（原直达 CatBoostError
  内部异常无引导）——先清洗或切编码，DAG/会话两路径同覆盖
- **基准任务落仓（#74）**：`examples/digits/`（sklearn 1797×64）+ `examples/titanic/`
  （同 schema 合成 891×12：Age ~20%/Cabin ~77%/Embarked 2 条缺失 + 妇女儿童优先相关
  结构，种子 1912 确定性）生成器 + 配套示例 YAML；零 CSV 落仓（sklearn 自带 ×3 + 合成 ×1）
- **`make e2e-bench` 三基准冒烟门（#75）**：iris 五族搜索 / digits MLP+scaler+XGB /
  titanic 全清洗链三树族 + iris/RF 分类会话最小链（B2 执行位）；判据 = 节点全绿 +
  指标下限（iris·digits f1≥0.90、titanic≥0.60，报告实测折），越限 FAIL 行 + 非零退出
  （`BENCH_FLOOR_BUMP` 越限自证）；实测 28s（预算 <5min）；发布前人肉门 = 先 `make e2e`
  再 `make e2e-bench`
- **工业场景基准三链落仓（#74/#75 范式扩展）**：`examples/bearing_rul/`（预测性维护
  RUL 回归：36 机组 × 150–560 循环 run-to-failure 仿真，振动/温度特征随退化进度非线性
  增长 + 载荷-寿命幂律，种子 1950）+ `examples/steel_defect/`（质检类不平衡二分类：
  3200×13，缺陷率 ~8%，表面统计主导可分结构，种子 1969）+ `examples/soft_sensor/`
  （软测量脏表回归：2600×12，传感器故障缺失 T3 ~8%/pressure ~4%/feed_temp ~2% +
  催化剂老化/床温共线结构，种子 1869）——生成器契约测试钉 schema/缺失/结构/确定性，
  YAML 自动入 #73 反漂门。`make e2e-bench` 扩六链：bearing RF/XGB 回归、steel onehot
  三树族 f1、soft fill_median 清洗→XGB/RF 回归；下限实测定折（数据 + machinery seed
  双确定性，逐次重跑逐位一致）= bearing MAE≤20/15 循环、steel f1≥0.75、soft
  MAE≤0.60/0.80 纯度点。bearing 整数 RUL（唯一值 <5% 且步长=1）被 `x_y` 自动推断判
  LabelEncoder 式分类——YAML 显式 `task: regressor` 钉住（整数计数型回归目标的推断
  盲区，`x_y.task` 显式参数的正当用例）

### BREAKING：节点解耦收口——predict/search_input 上游边化 + shap 族化（构建/引用分家）+ 副产物全走中间件

- **`predict` / `search_input` 的 `run_id` 参数退役（BREAKING）**：模型改经 `model` 参数边注入
  （model__* 组寻址单值取最近声明）——节点内不再耦合 model__loader 恢复逻辑，流里可自由
  组合任意 model__* 上游；compute 工具面零变化（run_id 经 PREDICT/INVERSE 模板内的
  model__loader 节点承载下传）；`mflowy.source_run_id` tag 改由 `ModelLoader.source_run_id`
  单源派生（`runs:/` uri 解析，内存态不落 tag）
- **shap 族化（BREAKING，#67 裸节点决定二次演进——组合优于耦合）**：`shap_explanation`
  裸键 → 族 `shap__explanation`（真算）+ `shap__load`（引用重建）；族返回契约 = 裸
  `shap.Explanation`（多分类 3D 透传；`y_name`/`categorical_features`/`classes` 以实例属性
  附记）；三个绘图消费方参数边改 `shap` **组边**（最近声明，explanation/load 同契约互换）
- **`shap__explanation(model, load)`——load 测试集面必选**：按模型 x_names 取列（缺列
  fail-loud）、超 100 行采样、x_y 边退役；解释面 = 终评 X_test 同源——explanation 工具从
  训练 run 的 `X_test.parquet` 副产物下载喂入（工件寻址住工具层，节点零 run_id 知识）
- **`shap__load(load: Iterator[DataFrame])`——collect + 列签名分区**：流内两个 `load__file`
  声明（测试集 + shap_values.parquet），按 `shap__` 前缀列自描述分区（values 宽表恰一份、
  测试集至多一份，歧义 fail-loud），重建 Explanation 与真算同形
- **shap_values 副产物（#66 契约 shap 面）**：`log_shap_explanation` 中间件随 run 落
  `shap_values.parquet`——宽表（`shap__`/`value__`/`cat__` 前缀列逐列同质 dtype，自含采样
  数据面；3D 按 `class` 列分块）
- **副产物全走中间件**：x_y 任务类型证据表从节点体迁 `log_task_evidence`（取数面 =
  `XySpec` 尾字段 `task_evidence`，三元组索引访问全兼容）；model 族 training 行为的
  holdout 三件套迁移记债（帧数据为函数局部量，涉 ModelLoader 契约扩展——#69）
- 词表：非归组区 = x_y/predict/search_input（split 已族化为 cv__split）；shap 族 = explanation/load

### BREAKING：词表身份收口——plot 族返回契约破裂修复（#67）

- **`shap_explanation` 裸节点化**（`@handler(namespace="plot")` → `@handler()`）：解释数据非图，
  寻址键 `plot__shap_explanation` → `shap_explanation`；summary/dependence/sample_waterfall
  消费参数边 `plot_shap_explanation` 同名同步（原单下划线在"组名索引 / node_key 索引"两边都
  不命中——任何流必选边不可达，explanation 工具模板必炸，此前无模板编译测试所以漏网）
- **`predict` / `search_input` 移出 model ns**（裸键 `predict` / `search_input`，原
  `model__predict` / `model__search_input`）：DataFrame 返回者不污染 model 组寻址——评估图
  收集边（`model_test_*` 内层 `Iterator[ModelLoader]`）与单值边契约恢复单一（model 组余下
  CAT/LGBM/MLP/RF/XGB/loader 全部返回 ModelLoader）
- explanation / predict / inverse_optimization 三份模板流键同步；判据 = 三条最小流编译 +
  三份模板渲染后 build（Builder 编译缝）+ 词表身份钉测试（plot ns 只有图、model ns 只有训练节点）

### BREAKING：数据指纹面收口——df 指纹 CAS 化（cleaning 域）+ x_y↔cv 链指纹全拆（划分身份 = run_id 引用）

- **`mflowy.data.fingerprint.fingerprint` 重构为 CAS（BREAKING）**：签名收窄
  `*dfs` → `df` 单帧；实现 = 规范序列化（`to_parquet(index=False)`）字节流的
  sha256 hex——与版本工件同参落盘 **CAS 闭合**（内存 df 指纹 = 盘上工件 sha）；
  manifest 双记收敛：`versions[].fingerprint` 键退役，版本溯源 =
  `artifact_sha256` 单键，重放对账经闭合性直对工件 sha，`_detect_all` 备忘键
  为指纹唯一存活消费（纯内存优化设施）；**行序进指纹**（重排 = 不同内容——
  重放须保行序，版本工件本就保序落盘）、行标签不进（`index=False`，重放侧
  索引差不误报）、零列帧退化为 `rows=n` 记账后哈希（格式统一 64 位 hex）；
  同环境（同 pandas/pyarrow 版本）守卫 + 血缘留痕、不承诺跨版本稳定（原契约一致）
- **x_y↔cv 链指纹全拆（BREAKING）**：物化守卫在现有布线下恒真（spec 与 x_y
  恒为同 run 同源；跨 run 复用走 cv.load 的 TrainSplit 直通，本就无指纹校验）
  ——`XySpec` 四元组收窄为 (X, y, task) 三元组、`xy_fingerprint` 与
  `SplitSpec.data_fingerprint` 退役、`materialize_split` 指纹校验块删除（物化 =
  groups 重建 + 索引派生）、split.json / manifest / run tag
  （`mflowy.split_fingerprint`）指纹面退役——划分身份唯一承担者 = split.pkl
  副产物的 run_id 引用（#66 无指纹重现假设；原「行序不敏感重放」语义随守卫
  一并退役，指纹消费侧只剩 cleaning 域）

### BREAKING：cross_validation 契约随族居住——mflowy.model.cross_validation → mflowy.model.cv.types

- **模块路径（BREAKING，仅直接 import）**：SplitSpec/X_idx/y_idx/X_y 契约与
  describe_xy/_materialize_groups 机器自 `mflowy.model.cross_validation` 移入
  `mflowy.model.cv.types`——契约与节点同族居住（#66 分家收官）；entry point 词表
  零变化（namespace 声明即身份——YAML/DAG/MCP 工具面无影响）
- **log_cv 延迟导入退役**：历史破环（cross_validation 持装饰器 ↔ log_cv 断言
  SplitSpec）自节点移居 cv/ 起已结构性消亡——TYPE_CHECKING 块与函数内导入升模块级，
  过时破环注释一并退役；mlflow/x_y/_split 的重载延迟导入保持
- 测试镜像随迁：`tests/test_cross_validation.py` → `tests/cv/test_cross_validation.py`

### BREAKING：modeling 终局协议与工件面收口——finish(pdf) 单动作 + 节点副产物 + 终训走 JobProvider + 光标单表 H（#66）

- **#66 光标单表 H（BREAKING）**：`s.h_space` 成员退役——``H[param]`` 唯一槽位
  （标量 = pin、dict = 空间收窄），set_hp/set_hp_space 同写 H，last-write-wins
  显式（decision_log note 记覆盖前后值——同参数不可能同时持有两份来源的值，参数
  来源冲突结构性消灭）；snapshots/manifest/观察面形状同步迁移
- **#66 u_full → u（BREAKING）**：终局效用从「多目标均值」改为**代表指标直取**
  （全套指标中取 scoring 代表指标，`terminal_u` 单源；与 u_inner 同尺）；会话恒
  **单目标任务**——`x_y` 的 `targets` 收窄为 `target`（多目标 plumbing 退役：推断
  冲突分支/逐列循环/跨目标均值），并增 `id_columns` 形参（物化期从 X 剔除——划分/
  搜索/终训/DAG 重放同一数据面，ID 泄漏结构性消失，会话侧二次剔除路径删除）；多
  目标建模在会话外**三层循环**（多目标 → 多模型 → 单目标任务会话，文献综述阶段输出
  每目标的模型候选）。ledger/board/verdict/skeleton/findings 全链 `u_full` 键改 `u`
- **#66 节点副产物契约 + cv 引用加载分家**：族节点 training 行为把 X_test/y_test/y_pred
  作为 parquet 副产物随 run（run_id 寻址、行位对齐——X_test 与 run 下 final 模型配对，
  SHAP 解释等下游按 run 引用测试集与模型）；物化划分 split.pkl 随 run 由
  **log_cv 中间件**落盘（`log_pickle` 直写 run 工件树——无 tmp 暂存无二次拷贝）；
  新增 **cv 族目录**（`mflowy/model/cv/`——`namespace="cv"`，节点键 `cv__split`/
  `cv__load`）：`cv.split` 收窄为纯参数构建（`from_run_id` 形参与指纹降级警告分支
  退役——签名即契约，参数来源冲突结构性不存在）、`cv.load` 按 run 引用加载
  split.pkl（缺失 fail-loud 带持有清单；消费 run 不重落副产物，holders 清单恒 =
  生产者）；`cross_validation.py` 瘦身为 SplitSpec 契约模块；下游迁 **cv namespace
  参数**——`train_split`/`model_test_*` 边的 `split` 参数改 `cv`（cv.split 物化 /
  cv.load 直通，与 load/clean 族组消费同构）；会话开局物化走**节点中间件链**
  （mlflow 窗口/log_cv 留痕/print 捕获——M7 通道纪律结构性修复），split run_id 进
  manifest 作划分身份
- **#66 开局入参合同（BREAKING）**：`cv_card` 收窄为 **cv.split 配置 yaml 明文**
  （键 ⊆ split 参数面，fail-loud 带可用键清单）；`modeling_session_start` 显式参数
  化——target/task/metric/candidates/max_steps/id_columns 显式命名参数 **> card
  （DomainCard 背景）分层缺省**，合并结果冻结进 manifest（dict 协议卡形态与
  target 静默 or 链退役）；compute 工具与 validate_modeling_steps 的
  `modeling_steps_yaml`（文件路径）改为 **`modeling_steps`（yaml 明文）**——MCP
  边界不假设共享文件系统，数据文件例外保持路径；task_dir 锚定（M10）改从 load 步
  数据文件目录解析，血缘指纹按文本内容哈希
- **#66 final_train_test 改走 modeling 工具**：渲染 per-combo 计划（cv.load
  from_run_id + 证据配置全 pin）→ JobProvider.modeling 跑 DAG（会话引擎同步语境经
  asyncio 桥调 async provider）→ u/metrics 从 DAGResult 回读（ModelLoader 新增
  `holdout` 快照面）→ 副产物指针经 run artifacts 清单解析落账（s 自描述）→
  combo 冻结。**终训即 plan 执行**：交付的 plan_yaml 就是那次终训本身（重放承诺与
  执行事实同一）；交付恒是搜索时刻**证据配置快照 ∪ θ\***（搜后改 pin 不进唯一一次
  holdout 读）；holdout findings 机器抽单源 `holdout_findings`（holdout_report 委托）
- **#66 素材面两工具（BREAKING 改名 + 新增）**：`taylor_diagram_report` 改名
  **`draw_taylor_diagram`**（本地分析类两站点——终训副产物 fan-in 画泰勒；
  `taylor_byproducts` 边函数（edges ns）断言各模型 y_test 逐位同一——不同 holdout
  的预测不进同一张对比图；只绘图不做报告）；新增 **`modeling_report_skeleton`**
  （tools/list 31→32——只读零副作用：表格从账本渲染（u_inner/θ*/u/**rank**——
  冻结面名次 + rank_consistency 出账）+ 图片清单 + 叙事槽样板）
- **#66 finish(pdf) 单动作（BREAKING）**：pdf 必填（preview 分支退役——素材面走
  draw_taylor_diagram/modeling_report_skeleton 两工具）；收编 = 复制进 deliverables
  （tmp/rename 原子落盘）+ verdict；缺文件 fail-loud；已收局同 pdf 幂等重导出、
  不同 pdf 报错（不可变更）；finish 不再画泰勒不再聚合素材（职责外移）
- **#66 findings 作用域 = combo 键**：propose 组装处过滤（规则与 LLM findings 同
  粒度）——菜单 rationale 引用的证据不来自另一个 FE 组合
- **#66 窗口纪律与剧本钉**：`hyperparameters_search` 无 active run 不再建孤儿
  nested run（objective 窗口判定一次，无窗口裸跑 trial 体——SDK 直调零 mlflow
  写入）；test_workspace 增 **M11 断言**（SKILL 反引号调用形引用 ⊆ tools/list ∪
  运行期节点/动作面词表——剧本改名漂移即测试红）

### BREAKING：modeling POMDP 升级收尾——回路状态机重构 + 输入面结构化 + 观察面与终局报告（#62/#63/#64/#34/#35/#36/#38）

- **#62 回路与状态机重构（BREAKING）**：s = 光标（m/H/h_space/seed/F）+ 账本（per
  (family,F) 条目，真源 = 盘上 sqlite study——append-only）+ history；动作面从「族名
  apply」改为**八动作面**：`select_model`/`set_hp`/`set_hp_space`/`set_seed`/`select_FE`
  （配置编辑——零步费零奖励）+ `run_hpo`/`final_train_test`（计算动作——吃步，预算 =
  实验数）+ `done`（finish 控制面）。`run_hpo` = **search-only** 增量搜索（sqlite study
  `load_if_exists`；增量判据 = 目标同一性：空间收窄/pin/加深继续、seed/scoring 变重开）+
  θ* 上 OOF 预测/图集/findings 挂 run（run_id 寻址），reward = **Δu_inner**（新键 =
  u_inner）；`final_train_test` = **先搜后门店** + holdout 终评恰一次 + **combo 冻结**
  （掐断用 test 反复挑参），reward = 0。rollback 只回退光标——账本与盘上 study 永不
  回退（#58 P1 根治）。mlflow 拿血缘认证不拿工件（run tag 补划分身份
  `mflowy.split_fingerprint`/test_size/group_by）；split.pkl 归会话层（sid 寻址）
- **#62 节点两行为派生（BREAKING）**：族节点按 `extract_search_spaces` 空否派生
  hyperparameters_search / training 两行为——参数含空间 → **只做 inner CV 搜索**（θ*
  证据，不终训不碰 holdout）；全 pin → 全量拟合 + holdout 终评 + loader。`evaluate_holdout`
  形参全族退役（no-peeking choke point 移交 final_train_test 动作）；`training()` 收敛为
  纯终训（param_space/n_trials 形参退役）；`.mdp_family` 增 `machinery`（会话直调的
  estimator 机器，`model/utils.make_machinery` 工厂从签名派生——缺省单一真源）。DAG
  用户经 modeling 工具传搜索空间的旧「搜索+终训一步走」形态改为：hyperparameters_search
  扫描 → 全 pin 走 modeling 终训
- **#62 应用层工具拆分（BREAKING）**：`modeling` 工具与 modeling.yaml.j2 拆为
  `hyperparameters_search` + `modeling` 两工具两模板（compute 4→5，JobProvider 协议/
  Local 实现同步）；tools/list 29→31（另含 `modeling_report_findings`/
  `taylor_diagram_report`）；`modeling_finish` 两段式（pdf=None preview / pdf=路径
  收编，见 #64）
- **#63 CVCard（BREAKING）**：`ProtocolCard` 全链改名 `CVCard`（session start 签名
  `protocol_card` → `cv_card`、工具 schema、skills/CONTEXT/docs 同步）；
  `candidate_families: [str]` → **结构化候选** `candidates: [{family, baseline,
  space_hints, seed}]`——baseline 恰一（propose 时第一）、hints ⊆ 族签名声明空间
  （标量 pin / 区间收窄——文献知识注入搜索起点）、每候选默认 seed（搜索期内冻结）；
  select_model 注入候选 hints 缺省（π 回合内可覆盖——卡是先验不是边界）
- **#35 diagnose 诊断族**：OOF 观察面四算子（`diagnose.py`：error_concentrated /
  slice_collapse / fold_std / complexity_edge——输出对齐 detect 的 ErrorReport 先例
  kind/severity/evidence/message）；run_hpo 后自动接线（零步数：findings 落账本 +
  run 响应浮出）；kind 词表 = #38 规则订阅面（含 overfit_gap——final_train_test 产出）
- **#36 LLM findings 回路**：`report_findings` 控制面 + `modeling_report_findings` 工具
  ——schema 校验 fail-loud（kind ⊆ 词表、severity ∈ [0,1]、未知键拒绝）、落诊进
  models[key]、决策日志审计；propose 对规则与 LLM findings **均匀聚合**（同一 fire 入口）
- **#64 OOF 观察面与终局报告**：run_hpo 出 **OOF 图集**（model_evaluation 图族指向 OOF
  源，挂 run 下 run_id 寻址）；**泰勒工具** `taylor_diagram_report`（本地分析类两站点：
  多 run holdout 终训预测 fan-in → (corr, σ, rmse) + 图落盘；用 holdout 不用 OOF——θ*
  带选择性乐观偏差）；**finish 两段式（BREAKING）**：`pdf=None` = preview（聚合产物
  清单 + 绘泰勒 + decision_log/board 素材——零副作用，宿主多模态读取）→ π（LLM 或
  人）撰写终局报告后 `pdf=...` = **收编进 deliverables**（复制 + tmp/rename 原子落盘；
  引擎不编译不验内容——报告是给人看的便当，数字权威面是 verdict/board）+ verdict
  （u_full/findings/rank）；交付报告不可变（重写须弃局重开）——引擎侧 LaTeX 编译与
  `conclusion=...`/无参收局形态退役
- **#34 evaluate 增 scoring 参数**：`scoring: MetricName | list[MetricName] | None`——
  None = 全套件（报告面），单值/列表 = 只算所声明子集；K 折 scorer 与 holdout-val
  trial 传单指标省算力（分类概率族计算大头）；子集不含概率指标时 y_proba 可缺省
- **#38 propose 规则引擎重写（BREAKING 菜单序）**：菜单 = findings 驱动层（`driver.rule.
  fire/prioritize`——记分牌 weight×severity 降序、同分 owner/kind 字典序，浮于顶部）+
  基座谓词机层（策展序：baseline 第一、未跑候选、终训交付、配置动作面）。fold_std →
  加深从无条件改 **std 门控**（无 `n_trials` 形参的族零加深题——gate 回归防线）；
  complexity_edge/overfit_gap → 正则收窄/pin（族自声明 `.regularizers`）；slice_collapse
  → 类别权重提示；error_concentrated → FE 换档（规则挂 x_transformer 词表节点）；
  **无 leader（S 无模型信息）时 FE 候选零出现**（引擎级 gate 非权重压序）；unconsumed
  findings 显式报告（无静默丢弃）；会话侧按 **(action, params) 键合并**（同动作同参 =
  同一建议的多重证据——rationale 并集、命中溯源并列 `co_hits`）

### Added（driver 规则引擎 SDK——#37 决议落地）

- **@rule 节点属性声明**：规则知识随词汇主人居住（`node.rule = {kind: fn}`，与 clean 域
  `.repair` 同构）——无独立注册桶、无 hatch 扩扫；`kind=None` 为探索题（severity 记 0）
- **三轴正交**：订阅（kind）/ 资格（gate——显式具名谓词读 ctx，未过进 `skipped` 缺席解释）/
  排序（记分牌 `weight×severity`，同分 owner/kind 字典序）——资格与权重互不代偿
- **fire/prioritize 门面**：`fire(findings, ctx, nodes)` 纯函数（节点集显式注入、零目录查询）
  → `FireResult(candidates, skipped, unconsumed)`（findings 消费闭环）；`prioritize(cands,
  overrides)` 委托人权重覆盖 + `_score` 分解进 evidence（排序可审计）
- 候选契约复用 `pomdp.verify_candidate` 单源；声明/挂载/输出 fail-loud 七路契约测试

### BREAKING：data 域评审落地——kind 分类学门守 + 检测正交化 + 观测去冗余（#51 阶段一 + B1）

- **kind 分类学闭合注册表 `KINDS`**（`detect/errors.py`）：detect↔clean 外键（`.repair` 声明键/记分牌权重键/算子产出）四方闭合契约测试钉住；第三方 kind 开放（0.05 兜底），内置面闭合——后续声明式校验的新词干经此门进入（#51·A1）
- **`detect.pattern` 正交三拆**：`detect.constant`（方差）/ `detect.stringly`（格式）/ `detect.id_like`（身份）独立成词，kind 词干零改动（外键稳定）；id_like 保留旧互斥语义（stringly 占比 ≥0.8 否决——数值语义列不是标识列）。**BREAKING**：`detect.pattern` 退役，词表经 `uv sync` 重建（#51·B1）
- **禁止采样裁定收敛**：`_stringly_probe` 的 `head(50)` 位置采样改**全列向量化** `str.extract`（前净后污的列不再漏报，附回归测试）；`clean_apply(dry_run)` **去检测化**——只留 diff 预览，样本上的 `report`/`quality_preview` 响应键移除（**BREAKING**：响应形状变更；真观测随正式 apply 的 s′ 免费自带）（#51·A5/B1）
- **观测通道 L1 状态同一性 memo**：`_detect_all` 同指纹（=同内容）不重跑全量检测——纯函数备忘录化，零近似零失效逻辑；apply-before 复用上步 after、propose 复用最近态、finish 幂等重入复用终态（start + 2×apply + propose + 2×finish 覆盖 3 状态恰 3 次计算，原 8 次）；容量 256 超限整体清空（#51·A4）

### BREAKING：driver 叙事重构——YAML 平铺有序列表 + 编译期血缘图（#52/#59）

- **YAML = 单键映射有序列表**（列表项 `{node_key: 配置体}`——键即身份；NodeConf 删 namespace/node_name 存储字段（派生自 `split_node_key`），`enabled` → `disabled` 正向键；disabled 节点全解析（停用即完整图配置））：`flow`/`branches` 嵌套与 `placeholder` 退役（拓扑序强制，前向引用 fail-loud）；边完全由签名机器对节点表解析（`edges` 词表① > 组名索引② > node_key 索引③），YAML 零边声明；多实例 collect 保序取全 / single 取末位（与旧 BFS 最近的漂移）；disabled 剪枝为必选依赖闭包（可选依赖剥除走 MISSING 签名默认值、collect 部分出局只收幸存成员；规则剪掉仍被必选消费的节点 fail-loud）；失败记账 output=repr(e)、DAGResult.status 增 partial、被剪节点计入 runs（status=skipped）
- **编译即校验**：`Builder.build(valid=)` 参数与 `DAG.valid()` 门退役——词表 miss / M1 撞名 / 必选边不可达 / 类型不兼容全部在增量编译期 fail-loud
- **stop_on_error 退役**：一切运行失败 = 剪枝血缘后代闭包、其余路径继续；被剪节点计入 runs（`status="skipped"`、无 run）；`DAGResult.status` 新增 `partial`（有失败且有完成），`error` 全量聚合
- **binding 结构重构**：`UpstreamMeta`（dataclass，`type` 三值 node/edge/namespace——词表投影即写 YAML 的索引）+ `Upstream(meta, expect, call, nodes)` 取代旧 `Upstream`/`Edge` NamedTuple；`NodeEntry.name` 改存 node_key、`bindings` → `upstream: list[UpstreamMeta]`（扁平 via 投影，`name == via` 边自指条目过滤）——list-nodes/get_node_info MCP 响应形状变更
- **剪枝单源 `lineage_closure`（两策略三调用点）**：传播策略由**生效相位**决定——构建期生效（disabled / structural_rules，意图）只穿必选边、可选消费者走签名默认值；运行期生效（失败，事故）无条件全闭包停播。`_disabled_closure`/`_descendants` 随之退役
- **StructuralRule 新契约**：`Callable[[Context], Context | None]`——编译后钩子可见血缘（`ctx.nexts`/`ctx.prevs` 只读视图）；从叶子反向 BFS 生效、剪即摘除（孤儿规则对上游二次生效）；返回 None = 剪枝且**沿必选边连带传播**（构建期 warning 点名，取代旧"剪掉仍被必选消费即 raise"契约——出局闭包内核兜底完成）；必须返回同一 ctx（替换即 fail-loud）
- **Context 播种式构造**：`get_node` 每节点恰一次，fn 与解析产物（direct/via/pairs/by_param）织入构造器——运行期**结构性**零目录查询（非 memoization 性质）；prevs→nexts 接线唯一居所在构造器；`_strip_and_rewire` 收编为 `Context.rewire`（出局剥除回归主人）；边内层依赖从 builder 私有 via 表改住 `ctx._via_upstreams`
- **目录键歧义 raise**（红线 1）：不同 `(ns, node)` 对折叠为同一 node_key（节点名含 `__`）从 warning 改为 fail-loud，报错带冲突双方
- **退役清单**：`DEFAULT_MAX_DEPTH`、`Context.resolve/reachable/pull_upstream`、`PreviousContextNotFoundError`、`iter_node_dicts`
- **目录表扁平化**：`discover()` 返回 `dict[node_key, EntryPoint]`（与 YAML 键/NodeEntry.name 同构——寻址单源贯穿）；分组视图（`list_all`/`namespace_nodes`）经 `split_node_key` 派生；跨 (ns, node) 键歧义（节点名含 `__`）装配期 warning
- **点式身份残留清零**：`NodeResult.node`、`skipped_result`、图键（`[node_key.node_N]`）、`mflowy.node` tag、M1 运行期防线地址全部 node_key 化；`NodeTable` ns 索引改 `split_node_key`（内核对 conf 派生属性读取归零）；外部 tag 消费方（`_parse_model_arg`、`search_experiment_model_run_ids` 的 LIKE 面与 module 提取、mcp `filter_steps`）联动迁移——**存量实验库的点式 tag 在新检索面 miss（同 run_id 引用不受影响）**
- **`bare_node_name` 退役**：node_key 的逆只剩 `split_node_key` 一份（其唯一消费方 mcp requires 键迁 `split_node_key(key)[1]`）
- **目录查询键收敛 node_key**：`get_node`/`entry`（门面与 `EntryPointCatalog`）改收单参 node_key——builder 零拆装（原 key→split→pair→key 往返消除）、Context 对 conf 派生属性最后一处读取消除；`get_node_info(namespace, node_name)` 对式保留为 MCP 外部契约（内部组键）；边查询令牌仍为裸边名（参数同名约定，内部组 `edges__` 键）
- **血缘去缓冲**：`Context._lineage` 运行期缓冲退役——`lineage` 改为 `_prevs` 派生只读视图（成功 run 下与实际消费恒等；失败 run 为声明超集，差集由异常文案指认）
- 影响：explanation 模板去 branches、10 个 examples 平铺迁移、model `prune_x_transformer_step` 血缘化、`log_df_diff` 改走 `upstream_contexts` 侧信道

### BREAKING：model split 契约收敛——SplitSpec = 划分器 + 数据指纹（#60）

- **SplitSpec 形态翻转**：`train_idx/test_idx` 索引数组与物化 `groups` 退役——契约成员改为 `outer`/`inner` 双划分器（对称、可 pickle，会话 split.pkl 序列化路径不变）+ `data_fingerprint`/`data_summary`/`n_rows` 数据血缘 + `group_by` 列声明；契约体积 O(n)→O(1)，索引/groups 全部是物化期派生物（确定性由划分器持种子保证：物化期派生 ≡ 构建期派生）
- **数据指纹单一真源（`mflowy.data.fingerprint.fingerprint`）**：行哈希聚合（`index=False` 对行序不敏感——同内容集重放不误报，守卫值内容集同一性；零列帧退化为行数记账），cleaning_session 版本指纹/检测备忘、x_y 节点（`XySpec` 四元组携带）、SplitSpec 物化守卫三方共用一个实现；split 层零重算只绑定，物化校验 = XySpec 携带指纹 vs spec 绑定值。与文件级 sha256 tag（`mflowy.data_sha256`，`load` 节点 fn 体内经 ContextVar 并入、中间件补写 run——文件字节粒度）为两个粒度的血缘并存
- **按输入形态分工落定（版本账本双记 + split.json 溯源）**：clean 各版本记录新增 `artifact_sha256`（版本 parquet 的 sha256——事后可对盘上工件独立复验；`fingerprint` 行内容指纹保留，备忘/重放对账两个语义消费方在用）；`split.json` 新增 `source_artifacts`（workflow_tags 的 `mflowy.data_*` 系——load 起点工件血缘随划分留痕，多 load 后缀齐全）
- **物化入口自封装三步**（`materialize_split`）：指纹校验（不符即 fail-loud 报两侧摘要——堵"同 shape 异数据静默错切"缺口；train_split 边函数天然获得该不变量）→ groups 重建（outer/inner 两处消费同源）→ 索引派生；`TrainSplit` 新增 `groups` 成员（train 行空间，inner CV / split_val 消费面）
- **train_ratio → test_size**（sklearn 对齐；缺省 0.2 ≡ 旧缺省 0.8）：split 签名尾部 `**_` catch-all 同步退役——旧名 pin 在调用边界 loud 失败而非静默吞默认；log_cv 构建期留痕改为 spec 所知（n_rows/test_size/fingerprint/group_by 进 split.json——数据血缘可观测）
- **`XySpec` 三元组 → 四元组**（`X, y, task, fingerprint`，NamedTuple 化）：x_y 节点在 dropna/类型收敛后的行内容上计算指纹并随产物携带——全部解包消费方（FE 节点 ×13/shap/session/split/edges）随迁；`materialize_split` 签名改 `(spec, x_y)`
- 影响：旧会话 split.pkl（旧形态 spec）不可载（会话为短生命周期工件，不做迁移）；examples/diabetes 参数名同步；训练侧消费 `spec.inner` / `split.groups`

### 版本号（并号回填）

- 0.1.2 之后仓库内曾以 1.0.0 → 2.0.0 → 3.0.0 记账，但均未发布（PyPI 实存仅 0.1.2）——
  两批 BREAKING 变更并号为单一 0.2.0：0.x 阶段 breaking 以 minor 表达

### Added（driver 工程质量：params 撞名双 fail-loud + 计划喂执行 + SessionStore 原子化）

- **M1 params 撞参数边名 fail-loud**：判定与文案单源在
  `binding.param_edge_names`/`raise_if_params_collide`——`DAG.valid()` 构建期拦截 +
  `handler_wrapper` 运行期防线（护住跳过 valid 的直跑路径）；四路契约测试进
  test_valid（撞名构建期拦截 / Annotated 名直通 / 运行期防线 / 零副作用保持）
- **M2 计划喂执行**：`ctx.resolve()` 计划 memo 扩为边清单+边表+缺失清单三产出，
  `_bind` 消费计划 memo、`handler`/`handler_raw` 实例缓存（链从 raw 属性派生，
  镜像 adapter 语义）——**run 期间零目录查询**（"现算热更新"赌注的扫描成本
  收敛到计划期独占）；`test_plan_feed` 以 adapter 调用计数钉住（含重复 run）
- **M3 driver 侧**：`SessionStore.write_manifest` 原子化（tmp+rename——manifest 是
  会话 commit 点，撕裂写不留半开状态）；`gc` 对不存在 root 返回空报告（显式
  root 参数路径无前置过滤，守卫进基座）；manifest 级 `is_terminal_state()`
  共享单源（两域 `_is_terminal` 委托随 #30/#31 合规迁移消逐字重复）
- `max_depth=20` 魔法数提为 `config.DEFAULT_MAX_DEPTH` 单源（context/edge 六处
  签名共用）；`NodeConf` 陈旧动态字段注释删除；driver README 七处漂移修复
  （`steps_to_yaml`→`flow_to_yaml`、失败定位改图键 `[ns.node.node_N]`、`verify_*`
  牙齿位置改域侧测试、builtin_plugins 包名/templates 位置/middlewares 路径、
  Plan-Execute 表补计划喂执行行）

### Changed（BREAKING — 目录分组：data/model 迁入 packages/builtin_plugins/，对 pip 用户零影响）

- `packages/data` → `packages/builtin_plugins/data`、`packages/model` →
  `packages/builtin_plugins/model`（git mv 保历史；distribution 名 / extras / entry
  points / wheel 产物零变化——破坏面仅限仓库内物理路径与开发者的 import 习惯）
- workspace members 收窄：显式平台包（utils/driver/mcp）+ `packages/builtin_plugins/*`
  glob——分组目录与 entry points 组 `mflowy.builtin_plugins` 机械对齐；分组目录内只允许
  distribution 目录（uv glob 对无 pyproject 子目录 fail-loud，实测）；组 README 上线
  （driver/mcp README 的既有链接就此成真）
- CONTEXT.md 词表修订：builtin_plugins 从整体禁用改为"禁指单包"（1.x 单发行版结构仍退役；
  词现仅作分组目录名与 entry points 组名）；域包词条补目录锚点
- 配套路径更新：根 pyproject per-file-ignores、tests/test_workspace.py 包路径映射
  （列表→路径表）、AGENTS.md 分层表/添加能力/文档路由、根 README 深读链接

### Changed（BREAKING — 子包结构收敛：plot 主题门面上移 utils + data_analysis 展平 + cross_validation 合并，entry point 词表零变化）

- `mflowy.data.plot.base`（56 行主题门面）+ `data/plot/__init__.py` 的图表环境就绪块
  （Agg 后端 + seaborn set_theme）合并上移为 `mflowy.utils.plot`；utils 新增 `[plot]`
  extra（numpy/matplotlib/seaborn），data/model 默认依赖 `mflowy-utils[plot]`——
  model 侧"主题就绪绕道 data.plot"的跨域表现层依赖就此消除（`model/plot/__init__`
  直接复用 utils.plot）；`from mflowy.data.plot import X` 包级兼容面保留，
  `mflowy.data.plot.base` 模块路径退役
- data 域 `plot/data_analysis/` 展平：7 图节点 + `_grid.py` 上移直住 `plot/`（单子女
  分组层纯开销；目录与 `plot` namespace 前缀精确对齐），测试镜像同步
- model 域 `cross_validation/`（split.py + types.py）合并为单文件
  `mflowy/model/cross_validation.py`（对齐 `x_y.py` 单文件族先例；节点身份本就是裸名
  `split`，目录不参与 entry point）；消费方 import 路径 `cross_validation.types/split`
  → `cross_validation`

### Changed（BREAKING — modeling 会话真决策化：SH 两段式取代相位机 + 族目录开放 + task_hint 退役）

- 建模会话动作面收敛为 `apply(node, params)`（与 clean_apply 同形）：model 族节点（run
  语义：搜索 → θ* + u_inner → 终训 → loader 落盘；响应显式 reward/state_digest）∪ FE 节点
  （pipeline 槽位语义，命中规则 = 模型已选定 + leader structure 非冗余）+ 控制面独立函数
  （rollback/finish/modeling_abandon 新工具）——P1-P3 相位机、edit/eval_mini/eval_full 词表、
  分档预算 (max_mini, b_full) 退役；预算 = 单一 max_steps（每动作吃一步，π 自行分配）；
  board 键 = (family, FE 组合)
- **资源机制全退（#33）**：资源（树数/epoch）退回普通签名参数（pin 即深，签名默认即预算）——
  .mdp_family 瘦身为目录标记 + structure、五节点与 training 的 resource 形参、partial→full
  升级、staged 资源曲线与 Hyperband 接线全部退役；早停（holdout-val 内建）是真省算力者；
  search() 降纯骨架 + 场景封装 search_ml（tpe+显式 NopPruner）/ search_inverse（单目标 tpe /
  多目标 NSGAII）/ search_dl（tpe+Hyperband 预设，暂无消费者）
- ProtocolCard：`banned_families`/`max_mini`/`max_full`/`resource_priors` 退役，新增
  `candidate_families`（research-review 文献候选收窄；阶段二 rank 保持深钻随资源退役一并退役）
- 族目录开放：`registry.families()` 硬编码五族 → discover() 过滤 `.mdp_family` 声明者
  （强制完整声明，未声明 = 对会话不可见，DAG 词表不受影响）
- 五个模型节点（XGB/LGBM/CAT/RF/MLP）删除 `**_`：未知/拼错参数从静默丢弃变为调用边界
  立即 TypeError（DAG 用户复制参数块须先清理）
- finish 一次结算多模型交付：deliver_keys 交付集选择、per-model terminal_utility/findings/
  可独立重放 plan_yaml（pin 常量 + θ* 全 pin）、rank 一致性诊断（u_inner vs u_full 秩相关
  ——inner/holdout 序的 SH 赌注出账）、verdict notes（error_concentrated 回流清洗提示）；
  accepted 恒真（test 读即终局）；重入幂等重导出（换交付集 loud 拒绝）
- `DomainCard.task_hint` 跨包退役（问卷问项与 clean 终局 U 的 hint 分支连坐）——task 唯一
  归属 `ProtocolCard.task`；存量卡传 task_hint 即 fail-loud 提示

### Changed（BREAKING — plot 词表裸名化：numeric_scale_box → violin / numeric_quality_kde_hist → histogram）

- `plot.numeric_scale_box` → `plot.violin`：横向 boxplot 重构为纵向 violin（inner="quart"
  分位线 + cut=0 不外推 + 均值菱形标记），共享 y 轴的跨特征量级/尺度对比定位不变；分位
  统计表返回值（whisker 界/outlier 计数与值）保留为精确读数侧车，outlier 计数标注移入
  x 轴刻度标签
- `plot.numeric_quality_kde_hist` → `plot.histogram`：实现不变纯改名——data 域 plot 收敛为
  "裸 chart 名（分布图）+ target_*_by_* 语法族（判读图）"两族，与 model 域命名一致
- `statistic.univariate` 参数 `top_k` → `category_top_k`（三义消歧：eda 工具 top_k=高相关
  对数、target_trend_by_numeric top_k=特征数）；内部事实出口 univariate_facts 连坐，
  输出字段 `top` 不变
- eda 模板在 univariate 后插入 plot.histogram / plot.violin 两步——单变量"统计+图"成对
  分组，零参数接线（默认全体数值列）
- data_profile docstring 漂移修正："统计摘要 + 分布图" → 实际步构成（形状/dtype/non-null）

### Changed（BREAKING — clean 动作族解耦：detect 与 clean 分离，词表 11→19 命名原子）

四层架构：判断归 detect（ErrorReport）/ 规则实现唯一居所 `detect/rules.py`（与修复同尺，
std ddof 口径分歧从事故变结构）/ 执行归命名原子 / 映射归 `.repair` v2。会话引擎的
转移/观测/守卫/预算机制零改动。旧 plan_yaml 重放断裂（映射见下）。

- **退役 7 动作 → 原子化居所**：
  - `fill_missing` → fill 家族 7 原子（fill_median / fill_mean / fill_mode / fill_const /
    fill_ffill / fill_bfill / fill_interpolate）——八策略级联（数值列请求 fill 静默回退
    median，动作对 (s,a) 撒谎）拆为教科书命名算子；dtype 不符即报错、无默认策略；
    指示列捆绑副作用独立为 `add_missing_flag`（投资动作：不改善 Φ，回报在终局 U，
    fail-loud 幂等）
  - `drop_missing`（名为删缺失实为按阈值删列）→ `drop_cols`（显式清单，阈值判断上移
    候选层）+ `drop_missing_rows`（行语义，新增）
  - `zscore_detector` / `iqr_detector`（检测+修复一体，与 detect 口径分裂）→
    `cap_iqr` / `cap_z`（winsorize 命名算子）+ `drop_outlier_iqr` / `drop_outlier_z` /
    `drop_outside`（显式界通用行过滤）；掩码自 `detect/rules.py` 同源 import
  - `variance_filter` / `correlation_filter` / `common_filter` → `drop_cols`（判断分别归
    pattern.constant / 新增 detect.collinear / 问卷确认；correlation 的删谁策略——留方差
    大者——居 `drop_cols.repair` 生成器）
- **detect 增量**：`detect.collinear` 新算子（kind=collinear，column=列序在前者、
  evidence 带 with/corr）；`detect.outliers` 增 rule 参数（zscore|iqr，缺省档位 3.0/1.5，
  evidence 带 fences）；`detect.pattern` 补类别常量列；scoring 补 collinear 权重 0.10；
  证据充实：missing 带 dtype、stringly 带众数后缀字符（strip_units 候选不再默认猜 "%"）
- **`.repair` v2 契约**：v1 单 kind 元组 → 多 kind 映射（drop_cols 一原子声明
  constant/id_like/collinear 三条修复）；生成器签名 (errors, card) → (errors, card, df)
  注入当前数据；候选自带 evidence（kind/列/计数/最重 severity）——决策有据；v1 元组与
  两参签名引擎永久兼容（第三方插件平滑面）；生成器适用性自筛按 `middlewares.type_family`
  类型族分区菜单（statistic/检测/原子适用性三家共用一把类型尺）；卡感知成契约
  （id_like 候选不点名问卷认领列）；**全员候选原则**——有 kind 对应的原子全部入菜单
  （fill_mean/interpolate/bfill、drop_outlier_z 均挂声明），菜单噪声由排序治理不由遮蔽
  治理：候选按 记分牌权重×severity 降序陈列（defect_priority 覆盖即时生效，置顶即默认档）；
  不挂声明的收敛为条件型动作 {python（逃生舱）, range_clip（域先验界，docstring 标注）}
- **组内填充**：fill_ffill / fill_bfill 增 by= 参数——组内前/后向填充不跨组泄漏
  （批量一致性场景：某批物理特征应一致时按批填充，不吸入上一批的值）
- **词形归一**：drop_duplicates → drop_duplicate（单数形，与 spec 词表一致）
- **观测面**：会话 `_detect_all` 接入 collinear（五算子）；data_profile/eda 模板补
  共线检测步（findings 编译层直达 agent）

### Changed（BREAKING — 嵌套研究流：P0 双侧退役 + propose_questions + 模板定稿）

- **P0 前向删除（双侧）**：data 侧 card_update/commit_card 元动作、card_partial 存储、
  phase 相位字段整体退役——`clean_session_start(card)` 唯一主通道直进动作面，start 即
  DomainCard.build 全量校验（原 commit 门居所迁移）+ never_drop 富化；model 侧 set_*/
  commit_protocol 谈判段同构退役——`modeling_session_start(protocol_card)` 唯一主通道。
  协议/问卷谈判上移剧本层（survey/model-brief skill 重写为嵌套协议：warm-up → eda →
  dive-in → clean_mdp）；术语 warm-up/dive-in 取代 P0
- **propose_questions 新工具（无状态）**：按 card_partial 缺项产出证据接地的下一问候选
  （必填优先、已答跳过、task_hint 随 target 出题），证据基质 = univariate 规则命中 +
  内部 detect.pattern（基质非动作、print 全吞——stdout 是 JSON-RPC 通道）；错列名当场报
- **模板定稿**：data_profile = load + statistic.profile（detect 五步与 plots 撤出）；
  eda = load + univariate + bivariate + plot 族（effect_size 与 detect 步撤出）；
  infer_task_type 工具与模板退役（task_hint 折叠进问卷生成器）
- **findings 编译层退役**：_compile_findings 与 DAGResult.findings 出口删除（模板无
  detect 步后恒 None 的死码）；kaggle 外环 analyze 的缺陷视图改道会话初始 report
- **无 target 流程中断不支持**（YAGNI，探索性需求出现再议）

### Changed（BREAKING — statistic 三分：观测面按"表→列→对"分层 + findings 溯源原则）

- **词表 2→4**：`statistic.profile` 瘦身为 df.info 薄封装（print-only、无 findings、无重
  schema 返回）；`statistic.univariate` 新增——统计规则按 type_family 分派（数值四分位/
  IQR/skew/kurt、类别 top-k 频次、时间跨度、全族基数比 is_id 判定共用 detect/rules.is_id_like），自带规则化 findings（missing 带率/空列/
  常量/id/skew 带值/outlier 远界提示带 min/max 锚——**无 severity**，喂出题不喂 Φ）；
  `statistic.bivariate` 新增——**无 target、全列两两配对**（target 无特判，先验地位属
  问卷卡），三路纯 dtype 分派（num×num→协方差+Pearson+Spearman；cat×cat→χ²+Cramér's V；
  num×cat→组间分布+t/Cohen's d 或 ANOVA/η²，分组 describe 迁移），跳过常量/高基数类别/
  时间列（清单进 print）
- **分档判据（效应量定档、p 把门、n 随行）**：strong/weak 需 p<0.05，效应量取 |r|-可比尺度
  （|ρ|/Cramér's V/√η²，强 ≥0.7 / 弱 [0.4,0.7)）；效应量达强档而 p 不达标对标 **unproven
  （未证）**——不占档不静默丢弃（小样本防伪 + 多重比较免疫）；top-effects 按效应量降序、
  p 破并列（取代 effect_size 时代最小 p 排序）
- **同尺升级**：`detect/rules.high_corr_pairs` 扩返回 (a, b, |r|, p)——collinear 的
  evidence 免费获得 p；bivariate 与 collinear 同源同值同 p（同尺断言钉住），规则居所的
  消费方扩为 detect/clean/statistic 三家
- `effect_size` 退役时序：随 #28（应用层）模板改写同批收口——模板仍引用，本批不删

### Changed（BREAKING — load 遥测隐式清洗退役：三黑盒动作白盒化进 POMDP 回路）

- `log_load_profile` 中间件删除（审计逃逸闭合：删全空行/删全空列/删常量列三动作曾
  只走 stderr logger、先于 detect 执行使 constant 缺陷在 DAG 侧结构性不可见）——
  load 族六 loader 摘链，**加载即原始数据**，DAG 数据形状变化（空行/空列/常量列保留）
- 白盒化落位：detect.missing 增行级 `empty_row` kind（全空行，evidence 带计数）；
  drop_missing_rows 增 empty_row 候选档（`{"how": "all"}` 零信息行清理）；
  drop_cols 增 missing 候选档（率≥80% 高缺失列整列删——兑现退役 drop_missing 时
  "阈值判断上移候选层"的迁移承诺，全空列 rate=1.0 自然入选）；常量列走既有 constant 档
- 中间件的内部执行顺序（先删空行再判常量）成为显式 POMDP 序贯两步：空行在场时
  nunique(dropna=False) 不认常量，删行后常量浮现——决策顺序从黑盒顺序变为
  agent 可见可回滚的步序
- scoring 补 empty_row 权重 0.10

### Changed（BREAKING — P0 内化：问卷段入住两会话，compile_survey 退役）

- 清洗会话 P0：`clean_session_start` 无卡开局 → card_update（增量校验填设）/
  commit_card（单向门全量校验 + 卡冻结进 P1）；记分牌谈判（defect_priority 权重随卡
  生效）；finish 载荷分层（report / deliverables）
- `compile_survey` 工具删除（问卷编译归 P0 元动作）

### Changed（BREAKING — 动作族目录统一动词词表）

- `loaders/` `cleaners/` `plots/` → `load/` `clean/` `plot/`（与 namespace 对齐；
  import 路径变化，镜像测试路径随迁）
- `clean/` 二级目录展平（族内扁平，与 load/statistic/detect/plot 一致）——六模块布局
  `drop.py / fill.py / cap.py（cap_iqr·cap_z·range_clip 同居）/ signal.py / strip_units.py /
  python_cleaner.py` + `_atom.py` 共享守卫；词表名不受文件移动影响

### Changed（BREAKING — driver 目录门面词表对齐）

- `mflowy.driver.discover.get_handler` → `get_node`（谈目录与图用「节点」，与 mcp 层
  `get_node_info` 命名对齐；`@handler` 装饰器不动——调度链视图仍称 handler）

### Changed（BREAKING — driver 目录 seam 退役）

- `Catalog` Protocol / `DictCatalog` / `use_catalog` / `active_catalog` / `has` 门面从公开面
  退役：间接层实证仅服务测试脚手架（共享 fixture `fake_plugins` 消费）；门面家族补齐
  `find_edge`/`entry`，委托唯一 adapter `EntryPointCatalog`（模块级单例，entry points 现算
  零缓存不变）；内核 9 处 + host 1 处直摸消费迁门面；测试伪目录移居根 conftest
  （`_FnCatalog` patch 单例位）

### Added（词表 miss 类型化 CatalogMiss）

- `CatalogMiss ⊂ ModuleNotFoundError`（携带 namespace/name/available）：「能力没装」与
  「装了但导不进」（ep.load 导入失败）按类型二分；探针 `find_edge` 坍缩为
  `except CatalogMiss`（metadata-first 特例退役）；三处消费收窄——extras 报告 miss 透传、
  清洗会话导入失败不再误诊为「不在词表」、节点枚举降级不吞 miss

### Changed（BREAKING — node_key 连接符 `_` → `__`）

- 寻址名（node_key）的组名⟷节点名连接符改双下划线：`plot_shap_explanation` →
  `plot__shap_explanation`。组名（`x_transformer`）与节点名（`shap_explanation`）皆可含
  `_`，单连接符不可逆拆分且跨层撞名（`x`+`transformer_foo` ≡ `x_transformer`+`foo`）。
  构造收敛单一来源 `config.node_key()`（四处调用：`NodeConf.node_key` / `@handler` 属性 /
  目录撞名检查镜像 / 节点信息裸桩回退）；裸名私有前导下划线统一剥除（`@handler` 侧
  原保留，与 NodeConf 对齐——目录层 `_` 词干本就不入词表）。YAML 为结构化字段免疫；
  全仓无拆分位点、无手写旧形数据依赖通道名

### Changed（BREAKING — model.auto 节点退役）

- **`model.auto` 删除（词表 65→64）**：内嵌 beam π 与「决策权放行 LLM」冲突——建模 POMDP
  以会话引擎（modeling_session）落地，beam 降级为参考策略的历史使命完成；其搜索机器
  （fast 配置表/结构组装/相容性）抽入 `mflowy.model.registry` 供会话与候选生成共用；
  直接测试随节点退役，相容性契约由 `test_registry.py` 承接

### Changed（BREAKING — tools 解耦：域包自持工具，mcp 退化为零域依赖 host）

- **域工具下沉**：数据域 9 工具（data_profile/eda/infer_task_type_by_statistic/compile_survey/clean_session_*）住 `mflowy.data.tools`（清洗会话引擎随迁 `mflowy.data.cleaning_session`、模板随迁）；建模域 5 工具（validate_modeling_steps/modeling/explanation/predict/inverse_optimization）住 `mflowy.model.tools`（JobProvider 协议+Local 实现随迁 `mflowy.model.job_provider`，`MFLOWY_JOB_PROVIDER` 语义不变；模板随迁）——**无装饰器、driver 零参与**，注册 = pyproject `[tool.mflowy.tools]` 手工声明（hatch 钩子并入 entry points 组 `mflowy.tools`，规避 PEP 621 动静冲突），镜像测试断言注册表 ≡ tools.py 公有函数
- **mcp host 泛化**：server/runner 经 `mflowy.tools` 组发现域工具（装哪个域包多哪批工具，`tools/list` = 已安装能力词表）；mcp 硬依赖收窄为 `mflowy-driver`（**零域依赖**，测试断言），tools.py 只剩平台工具（词表查询/mlflow 观测/工件读取/身份指纹）——第三方插件包可经 `mflowy.tools` 组自带 MCP 工具，零改 host
- 归位修正：`validate_modeling_steps` 实为建模域知识（消费建模模板与 step_options），随 model 域走
- lightning 落盘锚定（M10·#47）：MLP 训练 `lightning_logs` 目录改锚 task_dir（缺省兜底
  `~/.mflowy`）——仓库根/包根零运行产物（原 pytorch-lightning 缺省落 cwd 污染）

### Added（mcp 遥测端点 env 逃生舱（M9）+ CLI runner 通道契约收口）

- **遥测端点 `MFLOWY_TELEMETRY_ENDPOINT`**（M9·#46）：设备注册与 OTLP 导出共用同一解析
  ——env > 官方默认网关，自建部署（`docker/telemetry`）可被指向，与既有
  `MFLOWY_TELEMETRY=on|off` 运维逃生舱同层；agentcat 私有内部结构缺席（上游重构致
  monkeypatch 失配）时 feature-detect 显式告警而非静默失效——遥测退回官方导出面，
  工具 I/O 属性不可用随告警可知。
- **CLI runner（`mflowy <tool> '<json>'`）通道契约**：DAG 执行失败（`DAGResult.status`
  ≠ finished）退出码 1——原失败装在正常返回里静默 exit 0，e2e/CI 断链（SHAP 5/8 中止
  而 `make e2e` exit=0 的 prior bug）；参数非法 JSON / 非 JSON 对象退出码 2 带用法提示；
  info/查询类工具返回值回显 JSON（原一律不回显——查询通道经 runner 结构性不可用），
  建模类仍由节点 output 实时 print 上屏不重复回显。

### Changed（BREAKING — 域即包：mflowy-builtin-plugins 拆分为 mflowy-data / mflowy-model）

- **`mflowy-builtin-plugins` 退役，按 POMDP 域拆为两 distribution**（六 dist 锁步）：`mflowy-data`（数据域：loaders/statistic/detect/cleaners + EDA 图 + DomainCard/TASKTYPE——任务类型=target 列数据事实，随产生侧居住）与 `mflowy-model`（建模域：x_y/cross_validation/x_transformer/model + 评估/解释图）；依赖链 `utils ← driver ← data ← model ← mcp` 单向，**data 不得 import model**（测试断言）；共享件归上游（RANDOM_STATE→utils、plots 主题基座→data、middlewares 按消费面拆分）
- **extras 降格为域选择器，`[stats]`/`[modeling]` 退役**：`mflowy[data]`/`[model]`/`[mcp]`/`[deep]`；数据栈与 CPU 建模栈（sklearn/boosters/optuna/shap）成为域包**硬依赖**——装包即得完整域，裸装词表=已安装能力；**唯一可选 extra = model 包 `[deep]`**（torch/pytorch-lightning，MLP flavor，uvx 需 CPU 索引）；`mflowy[model]` 从此干净安装不碰 torch 索引；psycopg2-binary / iterative-stratification / numba / llvmlite 显式声明随批退役（全仓零 import：postgres 直连与 multilabel 切分为用户侧/未来需求，numba/llvmlite 由 shap 传递依赖）
- 两域包 entry points 共投 `mflowy.builtin_plugins` 组（组名不变，discover 零改动）；`list_nodes` 缺依赖降级面收缩至 `[deep]` 一档

### Changed（BREAKING — 入口形态重组：安装形态即用法）

- **命令入口重命名（``cmd`` → ``mflowy``，``mcpSrv`` → ``mflowy-mcp``，旧名移除）**：``mflowy <tool> '<json>'`` = CLI（JSON runner）、``mflowy-mcp`` = MCP server（main 内置 metadata 探测守卫：未装/<2.0 给安装提示而非 traceback）。脚本/文档中的旧名引用需同步（Makefile/examples/Dockerfile 已随本提交更新）
- **SDK 与遥测三依赖（``mcp>=2.0`` / agentcat / py-machineid）从 mflowy-mcp 硬依赖移入 ``[mcp]`` extra**：裸装即 CLI 零 SDK 拖带；既有「裸装 + ``mcpSrv``」客户端配置升级后需 ``mflowy[mcp,modeling]`` + ``mflowy-mcp``（server 缺 SDK 时 exit 1 + 安装提示）；runner 与直接 import 路径不受影响

### Added

- **P0 内化（两域会话，π 在 LLM 的完整形态）**：`clean_session_start(file_path)` /
  `modeling_session_start(final_path)` 无参开局（s0 = 裸数据/清洗终态）——问卷与协议
  谈判成为会话 P0 段的动作：`card_update`（增量校验填设）/ `commit_card` 与
  `set_target/set_task/set_cv/set_metric/set_constraints` / `commit_protocol`（单向门：
  全量校验进门 + 卡/协议冻结 + 建模域 D 物化与预算 P1 起算）；phase 改存储成员、只经门
  变迁（单调）；compile_survey / compile_model_brief 工具退役（校验逻辑进门，词表瘦身）
- **记分牌谈判（委托-代理启发落地）**：卡新增 `defect_priority`（kind→权重覆盖，
  per-episode 定 Φ）与 `min_row/col_retention`（保留率硬阈守卫——原 λ_r/λ_c 的谈判形态
  归还）；清洗终局 U 消费 `task_hint`（记分牌继承下游玩法）；`quality_score` 支持
  weights 覆盖——「先验→规则热更新」回路由 P0 每期谈判取代
- **finish 载荷分层**：两会话 finish 输出按受话人分 `report`（质量/终局 U/findings/步数
  ——代理对委托人的述职）与 `deliverables`（plan_yaml/final_path/loader——对下游的交付）
- **E2E 重链**：上传即开局全链（清洗 P0→回路→建模 P0→回路→计划重放）

- **建模会话（modeling_session，π 在 LLM）**：环境面状态机落 `mflowy.model.session`——
  P1 基线（baseline-first）→ P2 结构搜索（edit/eval_mini/rollback，分档预算 mini/full）→
  **eval_full 单向门**（test 唯一触碰，门后动作坍缩 {finish}，rollback 不可穿越）→
  P3 终局；board 只存 U_mini（no-peeking），决策日志可审计；finish 导出可独立重放的
  modeling_steps_yaml + loader + 终局 findings
- **ProtocolCard + compile_model_brief**：model-brief 问卷（规则预填+确认）→ 协议卡
  （task/metric/cv/预算/约束），四类相容校验 fail-loud（task↔target dtype、metric↔task、
  cv↔数据事实、预算下限）；协议 episode 内冻结（换协议=新 episode）
- **建模域 findings 词表（首批 3 kind）**：过拟合信号（inner−holdout gap）、切片塌陷
  （per-class recall）、误差集中列（|ρ| Spearman，回流清洗域载体）——eval_full 后规则
  引擎产出，随 finish 导出
- **MCP 工具 +6**：compile_model_brief / modeling_session_start / modeling_propose /
  modeling_apply / modeling_rollback / modeling_finish（注册表同步，镜像测试防漂移）
- **4-skill 齐**：model-brief（协议问卷）/ model（回合操作手册）随批，survey/clean 已就位
- **端到端验收**：三段式全链（survey 卡→清洗回合→model-brief→建模回合→finish）+
  计划重放（Builder 全量执行 = 会话终态同一语义）`tests/test_research_flow_e2e.py`

- **``list_nodes`` 缺依赖降级**：节点模块或其边探针 import 失败（缺 extra）时降级裸名 + warning——词表完整列出而非整查询崩掉（对齐目录命题「不静默消失」）；双入口守卫的契约测试随批
- **自主研究流（两个隔离 POMDP，π 在 LLM 的会话版）**：建档问卷 + 自主清洗 + 自主建模——
  - 清洗 POMDP 会话引擎 `packages/mcp/mflowy/mcp/cleaning_session.py`（环境面：转移/观测/
    奖励/预算，π 在调用方 LLM）+ 五工具 `clean_session_start` / `clean_propose` /
    `clean_apply`（dry_run 预览）/ `clean_rollback` / `clean_finish`：parquet 版本树快照、
    守卫（never_drop 列存活、目标列值禁改）、免费观测（每步 diff + 缺陷清单 + 质量分差分，
    检测算子是观测通道不占步数）、决策日志 (s,a,r,s′)、finish 导出可执行清洗计划
    （clean 词表 YAML 片段，可入 DAG 全量重放）；`clean.auto` 搜索版节点未落地——
    被会话版取代（决策权放行 LLM，见设计快照修订）
  - survey skill（`.agents/skills/survey/`，**非节点**）：两段式建档问卷（核心五问 +
    证据接地深问，"你不出清洗决策"）；`compile_survey` 工具把答案编译为 `DomainCard`
    （列引用/物理范围 fail-loud 校验）；`read_artifact` parquet token 友好钻取
  - `model.auto`：建模 POMDP 搜索版节点（beam：类别编码 × 数值缩放 × 模型族 + EVAL 两档——
    mini = 快速配置 inner-CV 不触 test 不进 mlflow；full = 委派真实 model 族节点全量训练，
    TPE + holdout 唯一触碰 + ModelLoader + 遥测全复用；建档卡 ID 列自动剔除防泄漏）——
    定位为参考策略，会话化（modeling_session）落地后退役，见 roadmap「两 POMDP 优化」
  - 新增 detect 缺陷算子族：`detect.missing` / `detect.duplicates` / `detect.outliers` /
    `detect.pattern`（stringly-numeric 单位污染），统一错误契约 `ErrorReport`
    （kind/位置/严重度/证据，行号采样封顶）+ `quality_score` 规则加权质量分（清洗 POMDP 的 R 侧）
  - 新增清洗动作节点：`clean.drop_duplicates`（按行去重，first/last/none）、
    `clean.range_clip`（物理范围裁剪，问卷 expected_ranges 消费端）
  - 端到端装配测试（会话清洗 → 冻结 → 建模 POMDP DAG）
    `packages/builtin_plugins/tests/autoflow/test_end_to_end_flow.py`；设计快照
    `docs/research/2026-10-09-cleaning-mdp-design.md` / `2026-10-09-modeling-mdp-design.md`

### Changed (BREAKING — 参数边 + @edge major：Plan-Execute）

- **签名即接线**：handler/edge 非 Annotated 命名参数 = 参数边——`Iterator[X]`/无注解 = 收集边（全量拉起 + `iter(list)` 物化迭代器，for 消费）；裸 `X`（含 `X | None`）= 单值边（最近命中 result 直注，签名免 `next()`）；空可选边**省略注入**，Python 签名默认值调用期原生生效
- **@edge 边函数**：具名变换边 `a-|edge1|->b`（`@edge def edge1(a: A) -> Edge1`）——无图位置、借消费方上下文晚绑定、边只吃节点（S3：跨边复用走函数体内直调）；首批 `train_split`/`x_preprocessors`/`model_test_*`；inject/getters（`middlewares/inject.py`·`getters.py`）处决
- **解析语义**：目录边名 > 组名 > 节点名三层寻址，全 miss 构建期 fail-raise；类型回退通道退役（不按返回注解匿名接线）；`collect_form`/`_unwrap` 单一谓词；撒谎 fail-loud 三处对称（节点输出/收集成员/边返回，`X | None` 下 None 合法）
- **Plan-Execute**：`dry_run` 旗标从原语退役——`ctx.resolve()` 记忆化产出静态边表（S5，快照语义），`DAG.valid`=EXPLAIN、`to_graph`/`__repr__` 图物化（mermaid `a -->|edge1| b` + nodes/edges JSON）、`run`=执行器；值级复核留执行器（语言天花板）
- **df 节点退役**：clean 族 9 节点、data_analysis 图 7 节点、`x_y` 直连 `(load, clean=None)`，statistic 族直连 `load`（统计画像看原始数据）；`df_source` 删除——词表零中转节点
- **ctx API**：`pull_upstream`（公开，yield-后-return 单值截断）+ `reachable`（dry 校验，`edge1.a` 前缀诊断）+ `resolve()`；`lineage` property
- **词表/工具正名**：`StepConf→NodeConf`（type/module→namespace/node_name、steps→flow）、`Workflow→DAG`、MCP `list_nodes`/`get_node_info`（签名式富化 `name(up1,up2=None)` 显依赖与可选性；`NodeInfo.upstream` 展平边函数保序去重；edges 桶不对外）；CLI 提示 step→ns
- **`Builder.build(valid=True)` 默认开启**；跨层撞名 discover warning；`XySpec` PEP 695 别名
- **示例 10/10 新词表**（type/module 全量转换）；掩盖性 skip（`_log_fold_metrics` 误标 PyTorch）修复，evaluate 测试恢复

### Changed

- **`x_names` 统一为 pipe 输入列名**：ModelPipeline 以整体作为模型单位（input X → output y），
  `x_names` 恒为原始输入列（CT/estimator 的 `feature_names_in_`），变换后特征空间不外露——
  修复带 ColumnTransformer（OHE 扩列）时 `search_input` 搜索空间与输入对齐的断裂；
  删除并存的 `input_features` 属性（双真相源）
- **K 折 trial 期间不再收集 eval 指标**：XGB/LGBM 的 fit override 仅在显式 eval_set 存在时挂
  eval（train-in-eval 只服务最终训练的 loss curve）；`run_study` K 折分支剥离
  `early_stopping_rounds`（xgboost ctor 早停在无 eval_set 时 fit 期 raise，跨 flavor trial 口径一致）
- **MCP `model` 参数严格裸 run_id**（flavor 全栈退役收尾）：旧 `flavor=run_id` 前缀格式由
  「兼容忽略」改为「明确拒绝」

### Added

- **`ctx.run()` 惰性子图执行**（driver）：`Context.run` 为单节点执行原语，返回本节点
  `NodeResult`（step/name/run_id/status/output）——pull 起来的节点有一等记账；
  `ctx.prev` 绑定期发现前驱未执行即惰性拉起（`prev.result` 哨兵恒等判定，多个消费者
  拉起同一上游仍只执行一次）——直接 `ctx.run()` 即以该节点为终点的按需子图执行：
  拉起发生在自身 run 窗口之外，不撞 MLflow 同线程活跃 run。执行语义定约：run 是执行
  请求不记忆（重复调用=重复执行+新 mlflow run，与 `Workflow.run` 两遍全量重跑对齐）；
  硬失败节点保持哨兵、后续可重试（非缓存失败）；失败节点以 Error 记账进 `runs`，
  `flow.error` 定位带 run_id；回显归入口层（workflow / 直接调用方），pull 子调用不回显
- **`@handler` 双平面 `inject` 槽**（插件 SDK）：`inject=` 关键字参数（单中间件或序列）
  注册数据面中间件（Get*/inject*），在 MLflow run 窗口外先行执行——链序
  `stop_on_error → inject → mlflow_log → *middlewares → fn`；`*middlewares` 照旧为
  遥测面（log_* 等依赖活跃 run）。旧式位置声明仍兼容（窗口内执行，仅不参与惰性拉起）；
  内置 51 处 `inject_*` 声明全部迁移至新槽
- **血缘 tag 缓冲落盘**（driver）：绑定期 prev 命中进 `Context._lineage` 缓冲，
  `mlflow_log` 收尾 `finally` 一次点分串接（保序去重，与 `append_tag` 直写同分隔符契约）
  写入 `mflowy.lineage`（原 `mflowy.input_steps`，随语义正名：记录被消费的直接上游
  节点集合，而非路径）——成功/失败 run 皆落盘；并把每次 prev 命中的
  get_run+set_tag 两次 tracking 往返折叠为每节点一次。窗口已关后的跨 run 追加
  （如 `GetSplit(model_ctx)`）仍走 `append_tag` 直写

### Changed

- **可视化固定 mermaid，`preview` 参数移除**（BREAKING，driver API）：
  `Workflow.__repr__` 固定输出 mermaid DAG 视图（节点按 `_next()` 调度序；
  边接入运行时血缘为进行中），name/tree 视图与 `Builder.build(preview=...)` /
  `Workflow(preview=...)` 参数移除——传参即 `TypeError`
- **driver 调度/执行分离**：Kahn 循环提取为 `Workflow._next` 生成器，单节点执行下沉
  `Context.run`（push 路径记账与行为不变）；`mflowy.utils.mlflow` 两处血缘 docstring
  去漂移（实现与测试均为点分，注释残留逗号表述）

- **load 步运行期数据指纹**：文件型 loader（csv/excel/parquet/python；file 按后缀委派继承）
  解析绝对路径后显式 `set_data_fingerprint`，`log_load_data_fingerprint` 中间件把更新后的
  workflow_tags 补写到 load 自身 run——modeling_yaml 内部 load 步的数据文件指纹全覆盖
  （多 load 步 `_n` 后缀，同文件幂等，远程引用跳过）
- **文件指纹 tags**：`Workflow.run(tags=...)` 经 ContextVar 注入、`mlflow_log` 应用到每个 node run——
  MCP 层为两类文件打指纹（`mflowy.{kind}_sha256` + `mflowy.{kind}_file`）：分析/predict/inverse 的
  数据文件（`data`）与 modeling 族的 `modeling_yaml`；`py:target` 引用哈希代码文件，远程/缺失
  引用静默跳过。指纹函数归位 `mflowy.utils.file`（`sha256_of`/`fingerprint_tags`）
- **pydantic 进 driver**（显式声明 `>=2.13.4`，driver 的真实必要依赖；其他组件经
  driver 间接获得，不再各自显式声明——mcp/builtin_plugins 原有声明移除）：
  类型转换知识挂在词汇主人侧——`DiscreteSpace.__get_pydantic_core_schema__`（yaml list→实例）
  与 `DualFormStrEnum`（枚举值/名双形式，`parse_enum` 的钩子版）；stdlib dataclass
  （`ContinuousSpace`）的 dict→实例 pydantic 原生支持

### Changed

- **科学性建模流程重构：holdout + inner-CV 单一最终模型**（BREAKING，训练产物契约）。
  对齐 sklearn 惯用法与 `research-flow` 既有纪律（CV 只用于选参，最终报告在未参与决策的
  held-out 上做一次）：
  - `cross_validate` 步九个模块退役，收敛为唯一 `split` handler（对齐 cross_val_score 的
    auto-selection：`cv`/`repeats`/`group_by` 三参数 + y 可分层性自动决策 splitter，
    `cv=null` 为 holdout-val 模式，`cv=样本数` 即 LOO、`cv=组数` 即 LOGO），产出
    `SplitSpec`（outer train/test 索引 + inner CV splitter + groups，同源同语义防组泄漏）；
    `folds.json` → `split.json`
  - 训练流程：可选 Optuna，评分按 inner cv 模式二选一——K 折模式走 `cross_val_score`
    （trial 无早停，折间 std 附带）；holdout-val 模式（`cv=null`）train 一次切 val、
    trial 手动 fit/predict/evaluate（早停可用，与最终训练同切分同口径）→ 最优超参全量重训单一 `ModelPipeline`
    （`Pipeline(preprocess?, estimator)`；早停模型从 train 内切 val，trial 阶段一律关早停）→
    holdout test 终评一次（test 不参与训练与选参）→ `mlflow.sklearn.log_model(name="final")`
  - 产物契约：`ModelLoader` 瘦身为「单一最终模型门面 + 身份」（三字段）；评估四件套
    （指标/预测长表/概率表/per-class 报告）为 mlflow-only 单向输出。fold 模型、fold 级
    run、fold 数一致性守卫、ensemble 聚合、`_FLAVOR_TO_WRAPPER`、wrapper `from_model`、
    `XPreprocessorsMixin`、`CatBoostPickleMixin.__getstate__`、autolog 全线删除
  - `loader`/`predict`/`search_input` 参数收敛为 `run_id` 单参，MCP `model` 参数同为裸
    `run_id` 严格校验（旧 `flavor=run_id` 前缀格式拒绝）；旧 fold 产物（`{flavor}_{i}` 命名模型）废弃，
    未命中 name='final' 即明确报错指引重训
  - 指标扩充：分类新增 balanced_accuracy/MCC/kappa/PR-AUC/Brier/ECE + per-class 报告表，
    proba 必需化（删 best-effort 降级）；`{scoring}_std`（best trial 折间分数）随 run 落盘
  - 评估图扩充：ROC / PR / 校准曲线（OvR 多分类）、Bland-Altman、混淆矩阵归一化
    （`normalize` 参数）+ per-class 报告表；多模型对比图改为同一 holdout test 上的现场重预测
    （公平性提升）；`x_names` 派生化（CT 的 `get_feature_names_out` / estimator
    `feature_names_in_`），`set_names` 只存 `y_names`
  - 旧 run（v0.1.x fold 产物）不再可复用：predict/explanation/inverse/resume 引用旧 run_id
    将 fail-loud 并指引重训

- **@handler 内嵌 validate_call：参数解析/校验移至调用期**（BREAKING，SDK 行为）：签名即约束的
  单一事实源——`Annotated[T, "描述"]` 旧约定兼容（字符串 metadata 被忽略按 T 校验）、
  `Field(gt=..)` 等约束在节点执行边界以 `ValidationError` 拦截（不再漏进函数体深处）、
  裸类型注入参数（DataFrame 等第三方类型）is-instance 直通。`params_phaser.py`、
  `StepConf.__post_init__` 的 `get_post_init` 转换钩子与 `@handler` 的 `params_phaser=`
  参数一并移除——第三方数据结构的 YAML 值转换挂 `__get_pydantic_core_schema__` 钩子
  （词汇主人侧）；discover 声明一致性校验由双属性收敛为单属性（`.handler`）。
  已接受代价：`StepConf.params` 保持原始 dict
  （mlflow log_params 与模板注入记录原始 YAML 形状）；类型/约束错误 fail-late（执行到该
  节点才报）。函数体内的参数判断分支批量迁移为签名声明（13 处/11 模块：correlation_filter
  threshold、variance_filter axis/threshold/column_thresholds、k_fold 族 n_splits/n_repeats/k、
  power/log_transformer 的 Literal 枚举、hash_encoder n_buckets、simple_cv 范围检查；
  数据依赖检查与跨字段约束（train+val<1、strategy='replace' 需 replace_value）保留函数体）
- **get_module_info 参数文档直出 JSON Schema**（BREAKING，MCP 契约）：`ParameterInfo`/
  `format_param_type` 退役，`ModuleInfo.parameters` → `parameters_schema`（标准 JSON Schema：
  properties/required/default/description/enum/$defs；旧式字符串描述桥接
  `Field(description=)`；Field 约束直出；实例默认值编码为结构化 JSON）——与 validate_call
  同一套 schema 机器，文档与校验永不漂移。字符串注解 eval 改 `typing.get_type_hints`
- **file_hash 工具收敛为 sha256 单算法**（定位=变更检查），返回形 `{path, sha256, size_bytes}`；`_flatten_params` 序列化兜底改用标准库 `asdict`（去 ContinuousSpace 特判）
- **深度扫描六批次**（组件依赖解耦 / 代码错位 / 同形收编）：
  - 删除死中间件 `log_X_y`（全仓零消费）
  - `format_param_type` 泛化：去 ContinuousSpace/DiscreteSpace 类名特判，dataclass/list 子类
    通用自描述（schema 输出逐位兼容）
  - `RANDOM_STATE` 迁 `builtin_plugins/constants.py`（utils/constants.py 删除）
  - driver 新增 `iter_step_dicts` 原始步骤遍历器（mcp 不再掏字典结构）
  - **builder_options 三层分家**：契约（`BuilderOption`/`StructuralRule` + Builder 注入点）在
    driver；prune/resume 工厂与 x_transformer 结构规则迁 `builtin_plugins/model/step_options.py`
    （loader 参数契约随词汇主人）；mcp 只做策略装配

- **`mflowy.utils.study` → `mflowy.model.model.study`**（BREAKING）：超参搜索空间与
  Optuna 编排归位 model 能力族——生产消费方 100% 在 model 域包，utils 回归纯通用工具；
  `mflowy-utils`/`mflowy-driver` 的 modeling extra（optuna）随之移除，optuna 终态为
  mflowy-model 域包硬依赖（`[modeling]` extra 已随 extras 降格整体退役，见上）

- **布局去 src 层 + 组件文档随包**：
  - 包内 `src/mflowy/<pkg>` 收敛为 `mflowy/<pkg>`（namespace wheel 与真 editable 行为不变，spike 验证）
  - 组件文档迁入 `packages/<pkg>/README.md`（随 wheel 分发）：driver = 技术架构文档（重写，含设计哲学谱系与架构图）、builtin_plugins = 能力目录 catalog、mcp = 工具三分 + JobProvider + 遥测（合并原 REMOTE_MODELING/TELEMETRY）、utils = 职责概览；`docs/` 仅留跨组件专题（research-flow / roadmap）

- **workspace 组件化：单包拆为五 distribution（uv workspace + PEP 420 namespace，BREAKING 内部结构）**：
  - `mflowy`（聚合包，PyPI 入口与 extras 透传不变）+ `mflowy-utils` / `mflowy-driver` / `mflowy-builtin-plugins` / `mflowy-mcp`，版本锁步（`tests/test_workspace.py` 断言）
  - `mflowy.compute` 更名 `mflowy.builtin_plugins`（包名 = entry point 组名）；`mflowy.middlewares` 拆解：`mlflow_log`/`stop_on_error` → `mflowy.driver.builtin_middleware`，`data_inject` 拆为 `middlewares/getters.py`（Get*）+ `middlewares/inject.py`（inject_*），`log_*` 与 `df_columns` 随迁 `mflowy.builtin_plugins.middlewares`
  - 测试随包迁移（`packages/<pkg>/tests/**`），根 `conftest.py` 全仓共享；新增 mlflow fluent run 栈隔离 fixture（修复 mock start_run 用例的跨包顺序污染）
  - `hatch_metadata.py` 随迁 builtin_plugins 并作为第三方插件参考实现；`Makefile build-whl` 改 `uv build --all`；Dockerfile 一次安装五 wheel
  - 依赖边界机器化：driver/utils/mcp 不得 import builtin_plugins（测试断言）；修复 `mlflow_log` 模块级 optuna import 导致缺 extra 环境无法加载插件的 latent bug

- **插件化架构：词表从 StepType 枚举改为 entry points 目录**（BREAKING）：
  - 删除 `driver/config.py: StepType`，`StepConf.type` 改为 `str`，YAML 值不变（load/clean/X_y/...）
  - `@handler` 删除首参 step（签名变为 `@handler(*middlewares)`），身份由 entry point name `step.module` 声明；`handler.py` 注册表 `_REGISTRY`/`_POST_INIT_REGISTRY` 删除，改为函数属性挂载（`.handler` 调度链 + `.convert_params` 转换器）
  - `discover.py` 重写：读取 `mflowy.builtin_plugins`（内置，`hatch_metadata.py` 构建期 AST 扫描生成）与 `mflowy.plugins`（第三方，安装即注册，后组覆盖前组）两组 entry points；目录查询零 import，加载惰性且坏声明 fail-loud——旧扫描机制“缺 extra 静默丢模块”问题随之解决
  - 第三方插件：以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]` 即可；镜像支持 `make build MFLOWY_EXTRA_MODULES="pkg==ver"` 定制插件包
  - 注意：editable 安装下新增 compute 模块需重跑 `uv sync` 刷新 entry points 元数据

### Added

- **file_hash MCP 工具**: 文件指纹（sha256 单算法，分块流式——上线前的三算法面
  已在同期收敛为 sha256，见上），供 agent 核验数据文件在阶段间未被改动

### Fixed

- **devcontainer**: 修复 Dockerfile 续行符缺失导致的解析失败与 `$APT_MIRROR`
  单引号不展开问题；`APT_MIRROR` 空默认（官方源）仅非空时替换，镜像源改由
  compose 显式传入；基础镜像对齐 3.12；移除仓库已不存在的 Tauri/Rust 依赖；
  恢复 `--no-install-recommends` 与 `~/.local/bin` 挂载点；`postCreateCommand`
  修正为 `make install`
- **`verify_output_hint` 重建于 `_unwrap`**（driver）：声明期校验与值级复核同域——别名/具体
  注解互认（修 required 别名边构建期误报）；可选边不再跳过声明校验（不匹配构建期
  fail-raise，而非运行期 `_check_value` 爆炸）；`optional` 形参退役
- **reachable/resolve 单次遍历收敛**（driver）：`ctx.resolve()` memo 升级双产物（静态边表 +
  缺失清单），`reachable` 消费同一 memo——C2 半落地清偿，声明校验随计划期一次完成
- **签名式节点串解析收编 `module.bare_name()` 单点**：mcp tools 与 catalog 测试 6 处
  `split("(",1)` 消费方统一，格式演化不再静默破坏解码
- **MCP `list_nodes` namespace 描述词表正名**：退役名 `X_y`/`cross_validate` 清出
  （x_y/split 已非归组，见空组）

## [0.1.2] - 2026-08-23

首个公开版本。（0.1.0 / 0.1.1 版本号已被 PyPI 文件名保留规则永久占用——曾上传的文件删除后文件名不可复用，故首发从 0.1.2 计。）

### Added

- **MCP-native 架构**: 全部能力以 pyfunc 工具暴露，MCP server（stdio）/ JSON
  runner（`cmd`）/ 直接 import 三入口；入口脚本定名 `cmd` / `mflowy-mcp`（避开
  mcp SDK 同名 CLI）；传输 stdio、身份经请求 `_meta` 透传
- **13 个 MCP 工具**: info（list_modules / get_module_info / validate_modeling_steps）、
  mlflow（list_runs / get_run / list_run_artifacts）、
  分析（data_profile / eda / infer_task_type_by_statistic，本地执行）、
  建模（modeling / explanation / predict / inverse_optimization，JobProvider 委派）
- **DAG 工作流引擎**: YAML 配置驱动（Jinja2 模板），`@handler` 装饰器自动注册，
  中间件链（数据注入 / mlflow 记录 / 错误即停），LIFO 深度优先拓扑调度，
  结构化 `WorkflowResult` 直传
- **依赖分层**: base / [stats] / [modeling] 三层 extras

### Changed

- **README 定位**: 自主数据分析/训练/预测 agent 的能力底座，Agent 自主回路
  （发现/组装/执行/读回）
- **协议卫生**: Taylor 图 vendor Copin taylorDiagram.py（public domain /
  CC BY 4.0），无 GPLv3 衍生代码
- **src 布局**: `src/mflowy/` 标准 src 布局，import 名 `mflowy`；
  Trusted Publisher CI 发布
