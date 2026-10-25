# 基准建模任务就绪度分析报告（快照式评估）

> 评估对象：MFlowy @ `3b91f861`（dev, 2026-10-24），版本 0.2.0-Unreleased
> 基准任务：Titanic（脏表二分类）、iris（干净多分类）、digits（64 特征多分类）
> 方法：能力词表盘点 + 三基准数据集**实跑**（compute 工具 DAG 路径 + MDP 会话路径）+ 全量测试 + 示例静态核对。本文为一次性评估快照，不随代码演进维护（防漂移豁免：评估产物，非常青文档）。

## 一、结论

**部分就绪。** DAG/compute 路径（YAML → modeling 工具）对三个基准全部跑通且产出完整交付工件，这是基准测试的主路径，可用。但存在三个阻塞性问题：

| # | 阻塞项 | 影响 | 定级 |
|---|--------|------|------|
| B1 | **MLP 节点全路径损坏**：handler 声明 `random_state`（默认 42 无条件传入），`MLPEstimator.__init__` 不收该参数 → 任何含 `model__MLP` 的 DAG 必炸 | 神经网络基线缺席，五模型对比降级为四 | P0 |
| B2 | **分类任务的 MDP 会话 `run_hpo` 全崩**：`_random_forest.py:31` 对 sklearn `cross_val_predict` 传入的 ndarray y 调 `.to_numpy()` | iris/digits/Titanic 这类分类基准走不了会话路径（回归会话正常） | P0 |
| B3 | **10/10 示例 YAML 漂移 + `make e2e` 断链**：示例仍用已退役的裸节点 `split`（现名 `cv__split`）与参数 `targets`（现名 `target`）；Makefile e2e 传已退役参数 `modeling_steps_yaml`（现名 `modeling_steps`）；示例 MLP 参数（n_trials/epochs）也已不存在 | 新用户/Agent 按示例复现必然失败，"三入口分发冒烟"本身不可用 | P0（对基准复现） |

结论一句话：**engine 核心可用、外围合同（词表改名后的示例/脚本收口）与两条 P0 节点级缺陷未收口前，不建议对外宣称基准就绪。**

## 二、实跑验证矩阵（全部真实执行，非静态推断）

| 路径 | iris（150×4 三分类） | digits（1797×64 十分类） | Titanic（891×11 二分类，脏表） |
|---|---|---|---|
| DAG modeling（搜索行为） | ✅ 7/7 节点，best_f1=**0.958**，~3min | ✅ XGB best_f1=**0.951**，8.5s；❌ MLP（B1） | ✅ 10/10 节点，三模型 f1≈0.64，15.7s |
| DAG modeling（全 pin 终训） | — | — | ✅ 8/8 节点，交付 12 项工件（ROC/PR/校准/混淆矩阵/classification_report/X_test/y_pred/test_probabilities/loss_curve…） |
| predict 工具 | — | — | ✅ 50 行预测落盘（须引用全 pin 终训 run；搜索行为 run 无 `final` 模型 → 引用即报错，报错文案清晰） |
| explanation（SHAP） | — | — | ✅ summary/dependence 图落盘 |
| MDP 会话 start | ✅ | ✅ | ✅ |
| MDP 会话 run_hpo | ❌ B2 | ❌ B2（同构推定） | ❌ B2 + CatBoost 拒绝类别列 NaN（`CatBoostError: cat_features must be ... NaN should be converted to string`） |
| MDP 会话（回归对照） | ✅ steps=1, reward=-0.78 | — | — |

全量测试：`uv run pytest` **1096 passed**（6:56）——套件健康，但上述 B1/B2 均未被覆盖（MLP 测试只钉签名/尾批契约不跑 DAG；会话测试全部走回归任务），印证"测试钉契约"红线下的覆盖缺口：**执行级（end-to-end）用例缺位**。

## 三、能力面评估（已具备）

- **词表完整**：clean 19 节点（填补 8 种/离群 4 种/去重/丢列/缺失标记）、detect 7、load 6（csv/excel/parquet/http/python/file 工厂）、x_transformer 13（onehot/label/ordinal/target/hash 编码 + 3 缩放器 + PCA/幂变换/分箱/交互）、plot 17（ROC/PR/校准/混淆矩阵/Taylor/SHAP 系）、model 5 族（XGB/LGBM/CAT/RF/MLP）+ loader。覆盖 Titanic 全部典型预处理需求（实测：drop_cols→fill_median→fill_mode→onehot→三模型）。
- **任务类型自动推断**：`x_y` 依 y 唯一值占比+步长判分类/回归，digits/iris/titanic 均判对（`classifier_evidence.parquet` 留痕可审计）。
- **防泄漏设计**：`id_columns` 从 X 结构性剔除（Titanic 的 PassengerId 实测剔除）；no-peeking（run_hpo 不碰 holdout）。
- **两行为派生**：超参默认即 Space → 搜索行为（Optuna 嵌套 trial run 全留痕）；全 pin → training 行为（final 模型 + holdout 终评 + 全套交付工件 + #66 副产物契约 X_test/y_test/y_pred 对齐落盘）。
- **遥测完整**：每步一 run、metrics（best_f1/std）、嵌套 trial、工件树齐整，mlflow 查询面可用。
- **降级路径**：`[model]` extra（无 torch）跑通全部树模型基准（iris/Titanic 实测均在无 `[deep]` 环境完成）——M6 所述"CPU-only 开不了会话"在开局面已不成立（会话可开、回归可跑；分类受 B2 阻塞）。

## 四、缺陷与问题清单（按证据）

### P0（阻塞基准）

1. **B1 MLP 套接不匹配**（`model/mlp.py:41` vs `model/_mlp.py:93`）：`make_machinery` 把 handler 签名默认（含 `random_state=42`）并入 `initial_params`，`MLPEstimator` 不收。进程内复现：`initial_params` 含 `random_state`。回归于 `78d4a089`（2026-10-19, #62 七票落地）。**任何数据、任何任务上都必炸**（默认值本身即毒）。
2. **B2 分类会话 OOF 崩溃**（`model/_random_forest.py:31`）：`y if y.shape[1] > 1 else y.to_numpy().ravel()` —— sklearn `cross_val_predict` 传 ndarray y 时 `.to_numpy` 不存在。DAG 路径 y 恒为 DataFrame 故幸免；会话 `run_hpo` 的 OOF 观察面走 `cross_val_predict` → 分类必炸。会话测试（`test_modeling_session.py::_hpo`）全部 `task="regression"`，未覆盖。
3. **B3 示例/e2e 全漂移**：10 个 `examples/*/*.yaml` 均含 `split:`（已更名 `cv__split`，CHANGELOG #66）与 `targets:`（已收窄 `target`，CHANGELOG #66 BREAKING）；Makefile `e2e-info`/`e2e` 传 `modeling_steps_yaml`（#66 已改 `modeling_steps` 明文）；示例 MLP 参数（n_trials/epochs）随 #62 退役。**`make e2e` 逐条核对当前全不可用。**

### P1（明显体验/合同缺陷）

4. **`load__python` 在 compute 工具下结构性不可用**：`_anchor_task_dir`（`job_provider/local.py:65`）只认 load 步 `params.file_path` 锚定 task_dir，而 csv/excel/parquet/python 加载器参数名是 `source`（仅 `load__file` 用 `file_path`）→ python 加载器无锚 → 实验创建时 `task_dir()/.mlruns` 即抛"请使用绝对路径"。**10 个示例全部使用 `load__python`**（叠加 B3 双重不可用）。规范路径实为 `load__file`（按后缀路由工厂，实测可用）。
5. **脏表进会话无守卫**：Titanic 原始 csv（类别列含 NaN）直连会话，CAT `run_hpo` 报错直达 catboost 内部异常，无引导性 fail-loud（应提示"类别列 NaN 需先清洗/转 str"或 select_FE 编码）。registry 已有 `cat_support` 声明位但未消费 NaN 语义。
6. **未知参数被静默吞**：load 节点 `**kwargs` 吞掉不认识的 `file_path`（实测 load__csv 只报缺 source 不报未知键）——违反红线 1（fail-loud）。本次实测还靠这个"假参数"才锚定了 task_dir，属于意外的隐性合同。

### P2（环境/工程化）

7. **本机默认追踪库 schema 污染**：`~/.mflowy/mlflow.db` 含当前 3.15.1 迁移链外 revision（`dc11669786a5`），`mlflow db upgrade` 无法定位 revision，库不可用。默认库为机器级共享路径，任何其他环境的 mlflow 写入都会毒化。基准验证需 `MLFLOW_TRACKING_URI` 指向隔离库（本报告实测方案）。
8. **搜索→终训语义门槛高**：树族核心超参默认即 Space（XGB 有 8 个），"全 pin"要求用户逐个写明全部超参才触发 training 行为/final 模型；漏一个就静默走搜索（实测第一次"终训"仍跑了 58 trials）。对基准工作流（search → best_params → 终训）可用但对新手是暗坑——建议 modeling 工具支持"从 experiment_id 搜索结果自动 pin best_params"的直通（docstring 已提及 resume 语义，未验证）。
9. **M12 未发布**：0.2.0-Unreleased 已堆 81+ commit 与多批 BREAKING，示例漂移（B3）正是"未按 release 边界回填用户文档"的直接后果。

## 五、最小修复清单（使三基准全绿）

1. `_mlp.py` `MLPEstimator.__init__` 增 `random_state` 形参（或 `make_machinery(MLP, ..., skip=("random_state",))` + seed 走 `seed_key`）——单行修复 + 补一条 DAG 级 MLP 执行测试。
2. `_random_forest.py:31` 改为 `np.asarray(y).ravel()` 等价 ndarray/DataFrame 双态写法 + 补分类会话 `run_hpo` 测试。
3. 批量更新 10 个示例 YAML（`cv__split`/`target`/MLP 参数/`load__file`）与 Makefile e2e 参数名，`make e2e` 纳入发布前人肉门（已有目标 M11/M12 佐证）。
4. `_anchor_task_dir` 增认 `source` 参数（或统一 load 族参数名），使 `load__python` 可锚。
5. （可选）会话开局对类别列 NaN 做 fail-loud 提示或自动 str 化。

前三项完成后：iris ✅ / digits ✅（含 MLP）/ Titanic ✅（DAG 路径），会话路径分类可用性随 2 恢复。

## 六、判定

- **DAG/compute 路径基准测试：可用（有条件）**——三基准实测跑通、指标合理、工件/遥测完整；条件 = 避开 MLP（B1）、走 `load__file`、终训需全 pin。
- **MDP 会话路径：分类不可用**（B2）、回归可用。
- **示例复现/新手路径：不可用**（B3）——对"基准测试就绪"而言这是最直接的否定项，因为基准测试的第一步通常是照示例跑。
- **综合：未就绪对外发布基准能力；完成五条最小修复（预计 1-2 天工作量）后可达到就绪。**
