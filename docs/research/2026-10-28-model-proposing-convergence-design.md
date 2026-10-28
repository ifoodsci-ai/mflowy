# model 域出题面收敛设计：问题体征分类学 + 规则表定义权归 propose

> 设计日期：2026-10-28。状态：**设计稿，未实现**。承接 #89 会话基座（姊妹篇：
> [2026-10-09-modeling-mdp-design.md](2026-10-09-modeling-mdp-design.md) /
> [2026-10-09-cleaning-mdp-design.md](2026-10-09-cleaning-mdp-design.md)）。
> 讨论裁定：data 域 #89 ③ 后出题纪律为「问题分类学 → 问题规则 → 处方菜单」；
> 本设计把 model 域出题面对齐到同一纪律。迁移表 M7 已随本讨论删除（执行面捕获
> 随编译执行构造化、观测面现状零裸 print——通道纪律由红线 4 常态承载）。

## 0. 定位：#89 之后 model 域出题面的四处残留

data 域菜单已是纯引擎出题（`propose_actions` = `fire(o) → poll_proposals() → return`，
cleaning_session.py:543-560）；model 域是两层拼接——findings 驱动层（引擎）+
基座谓词机层（session 手拼）。残留四处：

1. **基座手拼层**：`select_model` 候选 / `run_hpo` 首搜 / `final_train_test` / 配置
   编辑面以 `rule_key: "base.*"` 伪键手拼（不在 `engine.rules`，evidence `_rule`
   形状与驱动层不一致，`salience=-1` 垫底是两层拼接约定而非记分牌语义）；
2. **menu_ctx 间接层**：session 预消化词表（signatures / regularizers_of /
   has_n_trials / fe_slots_ok）注入 `state_digest.menu_ctx`——零外部消费方的
   私有握手（加一种规则 = 回头改 session 的 ctx 构造面；测试须手拼 ctx 形状），
   且 `propose_ctx` exit 写序（driver/session.py:214-215）把全量词表快照持久化进
   `observation.json`——环境常量上盘，浪费且带陈旧风险；
3. **尾巴三条**：5 个族函数各挂 `.structure` / `.regularizers` / `.mdp_family`——
   其中裸 `.structure` 无独立消费方（全部读取走 `["mdp_family"]["structure"]`，
   session.py:584/636/1322/1530），只是构造中间变量；
4. **分类学是代理指标**：`fold_std`（折间方差）/ `complexity_edge`（参数贴边界）/
   `overfit_gap`（inner vs holdout——语义混「CV 乐观偏差或分布漂移」）不是物理
   体征；`FERule` 把「换 FE」焊死为 `error_concentrated` 的唯一处方。

## 1. 核心裁定（三条原则，两域统一）

1. **出题知识住规则，词表知识住词表**。判据 = 单源红线：data 的 `.repair` 退役
   因为它是第二份问题知识（问题→动作映射挂在动作侧）；`families()` 保留因为族
   类型知识（machinery / structure / 正则面）全系统唯一一份。塞进规则 = 词表住
   进策略 = 造第二真源。
2. **`Rule.propose` 是规则表与查表逻辑的唯一定义处**。规则实例只持字符串键
   （可 pickle，`rules.pkl` 跨版本升级安全——该约束不区分「session 喂 ctx」与
   「规则自己查」）；表在 propose 内以词表为**环境常量**现查现建（import
   registry / fe_vocab）。静态字面量表 = `.repair` 复活，禁止。session 只做
   **状态投影**（观测函数本分）：findings / 光标 / `has_model` / `budget_left` /
   `has_cat` / 已训冻结摘要。
3. **问题视角出题**。分类学取物理体征级 kind；规则按问题 kind 注册，出处方
   菜单（候选动作 × 参数 × 档位理由，多动作类型）——data 域 ProblemRule 同构。
   先例：`compose_should_stop` 已证明「非 findings 驱动、状态谓词、垫底
   salience」的基座题可规则化。

## 2. 目标形态：三层职责分离

| 角色 | 职责 | 不再做什么 |
| --- | --- | --- |
| session（观测函数） | 状态投影：findings、光标、has_model、budget、has_cat、已训/冻结摘要 | 不预消化任何词表；menu_ctx 退役 |
| `Rule.propose` | 表构造 + 对 o 的查表；`families()`/`fe_vocab()` 当环境常量现查 | ——（唯一定义处） |
| registry（词表） | 原始声明聚合；`.mdp_family` 单尾巴承载全部族声明 | 不变（含 #58 准入校验、M6 降级） |

混合谓词拆解：现 `fe_slots_ok = structure × _has_cat` 拆为结构面（规则自取
`mdp_family.structure`）× 数据面（从 o 读 `has_cat`）。

## 3. 问题体征分类学

### 3.1 六体征与证据审计

| # | 体征 | 证据现状 | 缺口与成本 |
| --- | --- | --- | --- |
| ① | 欠拟合·训练/验证误差双高且接近 | 只有 u_inner 与 holdout u | **train-fold error**：CV 内对训练折 predict 即得，几乎免费 |
| ② | 欠拟合·损失曲线缓降/高位 | 曲线数据已在（`_build_loss_curve_df(train_metrics, …)`，_loss_curve.py:13）但只喂 plot | 投影进 findings + 形状判别（收敛值/斜率） |
| ③ | 欠拟合·学习曲线右侧相近且高位 | 无 | **净新增**（train-size sweep 多次拟合），居住面单独裁定 |
| ④ | 过拟合·验证误差显著高于训练 | 同① | 同一 train-fold error 缺口，补上即免费 |
| ⑤ | 过拟合·验证损失先降后升 | val 曲线数据在（同②） | 投影 + 形状判别（min 后上翘段） |
| ⑥ | 过拟合·学习曲线训练/验证明显间隙 | 无 | 同③ |

性价比断层：train-fold error + 曲线投影覆盖①②④⑤（改动小）；学习曲线（③⑥）
是真金白银新计算，居住面三选一裁定：run_hpo 附带一次性 / diagnose 工具面 /
暂缓。**族异构降级是设计内属性**：RF 无 epoch 曲线 → ②⑤对它不产 findings
（非缺陷，data 域 detect kind 按 dtype 族降级同款）；CAT/LGBM/XGB/MLP 可观测。

### 3.2 kind 设计

- **粒度 = 体征级**（`underfit_high_close` / `underfit_loss_plateau` /
  `overfit_val_loss_rise` / … 各自成 kind）——kind 是订阅面、记分牌权重、
  severity 归一化三者的单位；
- **severity 跨体征归一 [0,1]**（gap 比率 / 上翘斜率 / 间隙比），否则记分牌
  失去可比性；
- **现有 kind 去留**：`overfit_gap` 的 inner-vs-holdout 语义实为 CV 乐观偏差/
  漂移——收窄改名（如 `drift`）而非删除；`complexity_edge` 是参数位置代理，
  物理体征到位后降权或退役；`error_concentrated` / `slice_collapse` /
  `fold_std` 保留（本就是物理问题）。

## 4. 规则面设计

- **候选题规则**（退役 `base.select_model`）：读 `o.prior["candidates"]` ∩
  `families()`（现状 `_candidates`，session.py:144——注意是交集），策展序
  （baseline 第一 / 未跑在前 / 已训未全冻结末位）转为 salience/rationale；
- **问题规则**（翻转 FamilyRule/FERule 题型划分）：按 kind 注册，处方菜单 =
  候选动作×参数×档位理由。FE 换档成为处方的选项之一——欠拟合（信号不足）→
  {换 FE 加信号、加容量、减正则}；过拟合 → {加正则、减容量、换 FE 简化}。
  槽位资格门随处方走进各规则；`FERule` 作为独立题型消失；
- **调参题**：正则面（`regularizers`）与签名（Annotated Space 值域）规则内
  自取（import registry 现查），不再经 ctx 中转；
- **基座余项**（`base.run_hpo` 首搜 / `base.final_train_test` / 配置编辑面）
  同法规则化，或显式裁定保留并写明理由；
- **注册面与粒度**：注册时的词表迭代保留（哪些规则入引擎 = 词表的函数，
  `default_engine()` 现建）；粒度随题——候选题粗粒度（一条规则内含全表，
  即 `dict[family, dict[slot, bool]]` 形态）、问题题 per kind；
- **测试缝位转移**：手拼 ctx 喂假观测 → monkeypatch `registry.families()` /
  `fe_vocab()`（函数面可 patch，测试税不增）；
- **docstring 同步**：「规则纯消费 o」改「消费 o + 词表现查」——`fire(o)`
  契约与引擎本身不动。

## 5. 尾巴收敛（声明面单点化）

三尾巴 → 一条：`.mdp_family = {"structure": {…直书…}, "regularizers": (…),
"machinery": …, "fit_params": …}`。#58 准入键集扩 `regularizers`；
`regularizers_of` 改读 `mdp_family["regularizers"]`。声明载体仍是函数尾巴
（pyproject entry point 装载函数对象、属性免费随行——单注册站点携带节点与
域元数据，集中硬清单/模块级 dict/entry-point metadata 皆有第二站点或机制缺口）。

## 6. 落地序（三段 PR，前段不依赖后段）

1. **证据先行**（观察面富化，#89 ③ data 侧同款姿势，不碰规则）：train-fold
   error 进 CV 证据面 + 曲线数据投影进 findings。验收：findings 面含四新体征
   的回归测试；
2. **分类学扩容**：diagnose 算子族加体征算子 + `KNOWN_FINDING_KINDS` 扩 +
   severity 归一 + 记分牌权重；学习曲线居住面裁定。验收：kind 词表与订阅
   检查绿（unconsumed findings 无静默丢弃）；
3. **规则翻转**（合并全部机制项）：问题规则处方菜单 + 候选题入规则 +
   menu_ctx 退役 + 尾巴三并一 + 基座手拼层清零。验收：`propose_actions` 退成
   「状态投影 + fire + poll + 合并」；`base.*` 伪键清零；`observation.json`
   无词表快照；菜单形状回归锁策展序（baseline 序/降级末位不变）。

## 7. 不变项（明确不动）

- `families()` 词表与动作校验面：`select_model` 的 `family ∈ _candidates`
  fail-loud 门（session.py:483-485）不动；
- machinery / fit_params 可执行构造知识留词表——`Activation` 的声明性 params
  装不下 callable 工厂；
- 壳机制（六操作上下文 / 三文件写序 / 撕裂检测）、`fire(o)` 契约不动；
- data 域不动（本设计纯 model 域出题面）；
- 两域对称性表述：出题知识住规则、词表知识住词表——data 侧 KINDS/DomainCard/
  节点目录与 model 侧 families()/fe_vocab() 是同一纪律的两面。
