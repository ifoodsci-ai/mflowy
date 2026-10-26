# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

### BREAKING：规则引擎重构——散落规则聚合到实例（Rule/Activation/RuleEngine + open_engine 落盘），两域 MDP 统一 fire/poll 入口

- **实例门面取代节点属性挂载**：规则 = 实现 `driver.rule.Rule` 协议的对象（`key` 身份 +
  `propose(R) -> list[Activation]`），经 `RuleEngine.register(*rules)` 聚合；`fire(*keys, R)`
  按 key 激活（空 = 全部，未知 key fail-loud 带候选清单）产出 `Activation`（proposal 恰含
  action/params + salience 记分牌 + rationale/evidence 解释面）在实例内累积，
  `poll_proposals()` 按 salience 降序（同分 rule_key/action/params 字典序）聚合排空——
  连续 fire 一次取走，菜单确定性不变
- **实例落盘**：pkl 承载注册面（含运行期注册的自定义规则）与未 poll 激活态；
  `open_engine(path, default=域默认实例)` 盘上有则加载（损坏 fail-loud 带路径，不静默重建）、
  无则从默认实例复制（激活态清空起步）；with 正常退出原子落盘（走
  `mflowy.utils.file.write_bytes_atomic` 单源原语）、异常不落盘（失败调用零副作用）。
  规则须可 pickle——域规则 = 模块级 frozen dataclass 只持 str/num，词表/注册表对象在
  propose 内现查
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

### BREAKING：modeling MDP 升级收尾——回路状态机重构 + 输入面结构化 + 观察面与终局报告（#62/#63/#64/#34/#35/#36/#38）

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
- 候选契约复用 `mdp_session.verify_candidate` 单源；声明/挂载/输出 fail-loud 七路契约测试

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

### Changed（BREAKING — load 遥测隐式清洗退役：三黑盒动作白盒化进 MDP 回路）

- `log_load_profile` 中间件删除（审计逃逸闭合：删全空行/删全空列/删常量列三动作曾
  只走 stderr logger、先于 detect 执行使 constant 缺陷在 DAG 侧结构性不可见）——
  load 族六 loader 摘链，**加载即原始数据**，DAG 数据形状变化（空行/空列/常量列保留）
- 白盒化落位：detect.missing 增行级 `empty_row` kind（全空行，evidence 带计数）；
  drop_missing_rows 增 empty_row 候选档（`{"how": "all"}` 零信息行清理）；
  drop_cols 增 missing 候选档（率≥80% 高缺失列整列删——兑现退役 drop_missing 时
  "阈值判断上移候选层"的迁移承诺，全空列 rate=1.0 自然入选）；常量列走既有 constant 档
- 中间件的内部执行顺序（先删空行再判常量）成为显式 MDP 序贯两步：空行在场时
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

- **`model.auto` 删除（词表 65→64）**：内嵌 beam π 与「决策权放行 LLM」冲突——建模 MDP
  以会话引擎（modeling_session）落地，beam 降级为参考策略的历史使命完成；其搜索机器
  （fast 配置表/结构组装/相容性）抽入 `mflowy.model.registry` 供会话与候选生成共用；
  直接测试随节点退役，相容性契约由 `test_registry.py` 承接

### Changed（BREAKING — tools 解耦：域包自持工具，mcp 退化为零域依赖 host）

- **域工具下沉**：数据域 9 工具（data_profile/eda/infer_task_type_by_statistic/compile_survey/clean_session_*）住 `mflowy.data.tools`（清洗会话引擎随迁 `mflowy.data.cleaning_session`、模板随迁）；建模域 5 工具（validate_modeling_steps/modeling/explanation/predict/inverse_optimization）住 `mflowy.model.tools`（JobProvider 协议+Local 实现随迁 `mflowy.model.job_provider`，`MFLOWY_JOB_PROVIDER` 语义不变；模板随迁）——**无装饰器、driver 零参与**，注册 = pyproject `[tool.mflowy.tools]` 手工声明（hatch 钩子并入 entry points 组 `mflowy.tools`，规避 PEP 621 动静冲突），镜像测试断言注册表 ≡ tools.py 公有函数
- **mcp host 泛化**：server/runner 经 `mflowy.tools` 组发现域工具（装哪个域包多哪批工具，`tools/list` = 已安装能力词表）；mcp 硬依赖收窄为 `mflowy-driver`（**零域依赖**，测试断言），tools.py 只剩平台工具（词表查询/mlflow 观测/工件读取/身份指纹）——第三方插件包可经 `mflowy.tools` 组自带 MCP 工具，零改 host
- 归位修正：`validate_modeling_steps` 实为建模域知识（消费建模模板与 step_options），随 model 域走

### Changed（BREAKING — 域即包：mflowy-builtin-plugins 拆分为 mflowy-data / mflowy-model）

- **`mflowy-builtin-plugins` 退役，按 MDP 域拆为两 distribution**（六 dist 锁步）：`mflowy-data`（数据域：loaders/statistic/detect/cleaners + EDA 图 + DomainCard/TASKTYPE——任务类型=target 列数据事实，随产生侧居住）与 `mflowy-model`（建模域：x_y/cross_validation/x_transformer/model + 评估/解释图）；依赖链 `utils ← driver ← data ← model ← mcp` 单向，**data 不得 import model**（测试断言）；共享件归上游（RANDOM_STATE→utils、plots 主题基座→data、middlewares 按消费面拆分）
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
- **自主研究流（两个隔离 MDP，π 在 LLM 的会话版）**：建档问卷 + 自主清洗 + 自主建模——
  - 清洗 MDP 会话引擎 `packages/mcp/mflowy/mcp/cleaning_session.py`（环境面：转移/观测/
    奖励/预算，π 在调用方 LLM）+ 五工具 `clean_session_start` / `clean_propose` /
    `clean_apply`（dry_run 预览）/ `clean_rollback` / `clean_finish`：parquet 版本树快照、
    守卫（never_drop 列存活、目标列值禁改）、免费观测（每步 diff + 缺陷清单 + 质量分差分，
    检测算子是观测通道不占步数）、决策日志 (s,a,r,s′)、finish 导出可执行清洗计划
    （clean 词表 YAML 片段，可入 DAG 全量重放）；`clean.auto` 搜索版节点未落地——
    被会话版取代（决策权放行 LLM，见设计快照修订）
  - survey skill（`.agents/skills/survey/`，**非节点**）：两段式建档问卷（核心五问 +
    证据接地深问，"你不出清洗决策"）；`compile_survey` 工具把答案编译为 `DomainCard`
    （列引用/物理范围 fail-loud 校验）；`read_artifact` parquet token 友好钻取
  - `model.auto`：建模 MDP 搜索版节点（beam：类别编码 × 数值缩放 × 模型族 + EVAL 两档——
    mini = 快速配置 inner-CV 不触 test 不进 mlflow；full = 委派真实 model 族节点全量训练，
    TPE + holdout 唯一触碰 + ModelLoader + 遥测全复用；建档卡 ID 列自动剔除防泄漏）——
    定位为参考策略，会话化（modeling_session）落地后退役，见 roadmap「两 MDP 优化」
  - 新增 detect 缺陷算子族：`detect.missing` / `detect.duplicates` / `detect.outliers` /
    `detect.pattern`（stringly-numeric 单位污染），统一错误契约 `ErrorReport`
    （kind/位置/严重度/证据，行号采样封顶）+ `quality_score` 规则加权质量分（清洗 MDP 的 R 侧）
  - 新增清洗动作节点：`clean.drop_duplicates`（按行去重，first/last/none）、
    `clean.range_clip`（物理范围裁剪，问卷 expected_ranges 消费端）
  - 端到端装配测试（会话清洗 → 冻结 → 建模 MDP DAG）
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

- **`mflowy.utils.study` → `mflowy.builtin_plugins.model.study`**（BREAKING）：超参搜索空间与
  Optuna 编排归位 model 能力族——生产消费方 100% 在 builtin_plugins，utils 回归纯通用工具；
  `mflowy-utils`/`mflowy-driver` 的 modeling extra（optuna）随之移除，optuna 唯一归宿为
  `mflowy-builtin-plugins[modeling]`

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

- **file_hash MCP 工具**: 文件指纹（sha256/md5/sha1，分块流式），供 agent 核验
  数据文件在阶段间未被改动

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
