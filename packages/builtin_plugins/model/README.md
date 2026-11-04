# mflowy-model（建模域）

MFlowy 建模域包：建模训练 POMDP 的动作域——x_y / cv / x_transformer / model /
plot(model_evaluation + model_interpretability)。CPU 建模栈为硬依赖；torch 栈在 `[deep]`
extra（uvx 安装需 `--index https://download.pytorch.org/whl/cpu`，仅 MLP flavor 需要——
缺它时 `list-nodes` 降级标注，不影响其余能力）。

## 会话 POMDP 元素表（O/S/A/P·T/R）

契约下限与校验器住 `driver/pomdp.py`（Observation / FinishVerdict / Candidate）；本域语义如下（引擎 = `session.py`，s 元组与八动作面的完整定义见
模块 docstring。转移确定性——无概率转移表；可观测性 = diagnose/规则覆盖面：board 聚合与
findings 是真态的多对一投影——同聚合不同病灶不可分；u_inner/std 是隐藏真值的带噪估计
（fold_std 观测即为此设）；report_findings（#36）= π 自带传感器的私有观测落账，补程序化观测
函数的覆盖缺口）：

| 元素          | 本域定义                                                                                   | 居所                                        |
| ------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------- |
| O 观测        | `_observe` 域加字段（candidates / cursor / board / budget）——簿记态全可见；OOF 观察面（见下节）随 run_hpo 免费出，病灶可见性以 diagnose/规则覆盖面为界 | `_observe` / `diagnose.py` / `report.py`    |
| S 状态        | 真态 = s 元组（光标 m/H/seed/F, 账本 board, oof, budget, done, history）+ 未探 combo 的真值景观（不可直读，仅经带噪 u_inner 间接观测）；真源 = 盘上 sqlite study（append-only），账本条目为派生缓存 | 模块 docstring / `_board` / `_digest`       |
| A 动作        | 八动作面：配置编辑 ×5（零步费零奖励）∪ 计算动作 ×2（吃步）∪ done（= finish 独立入口）；词表外 fail-loud 带清单 | `ACTIONS` / `COMPUTE_ACTIONS` / `apply_action` |
| P·T 转移      | 光标编辑推快照；run_hpo 增量追加 study（目标同一性判据）；final_train_test 冻结 combo（holdout 读恰一次）；rollback 只回退光标——账本 / 预算 / 盘上 study 永不回退 | `_select_*` / `_run_hpo` / `_final_train_test` / `rollback` |
| R 奖励        | run_hpo = Δu_inner（新键 = u_inner 绝对值）；配置编辑与终训 r = 0；终局 = verdict（holdout u + rank 一致性，accepted 恒真） | `_run_hpo` / `finish_session`               |

- 动作族：`model/`（XGB/LGBM/CAT/RF/MLP/loader——全部返回 ModelLoader，组寻址契约单一）、
  `x_transformer/`（编码 ×5 + 数值变换 ×8）、`cv/`（types.py 契约叶子 + split 参数构建 + load 引用加载——契约与节点同族居住）、
  `plot/model_interpretability/shap/`（shap 族：explanation 真算留痕 + load 引用重建——构建/引用分家，
  契约 = 裸 shap.Explanation + 属性注记，消费方走 `shap` 组边）、
  非归组 `x_y.py`（任务目标划分 + 任务类型推理）/ `model/predict.py` + `model/search_input.py`（model 上游边消费，零 run_id 耦合）
- 共享件：`middlewares/`（log_cv / log_prediction / log_search_input / log_shap_explanation / log_task_evidence——
  副产物统一走中间件落 mlflow 工件）、`plot/`（评估/解释图，主题门面复用 `mflowy.utils.plot`）
- 类型：`model/types.py`（MetricName / ModelPipeline / ModelLoader；TASKTYPE 再导出自 data）
- 标注：`extras.py`（`model/mlp → [deep]`，0.2.0 起唯一可选 extra）
- 判据：镜像路径有对应测试（`tests/`），`uv sync` 后 `list-nodes` 可见（新模块须重跑 `uv sync`）

## 族节点两行为派生（#62：extract_search_spaces 空否即判据）

- 参数含搜索空间 → **hyperparameters_search 行为**：inner CV 选参（θ*/u_inner 证据），
  不终训不碰 holdout——配 `hyperparameters_search` compute 工具做候选扫描
- 全 pin（空间空）→ **training 行为**：全量拟合 + holdout 终评 + loader
  （`modeling` 工具 / plan 重放 / resume 的生产路径）
- 共享底座 = `_pipeline.hyperparameters_search` + `_pipeline.training`（estimator 机器
  由 `model/utils.make_machinery` 工厂从签名派生——缺省超参单一真源 = 签名）

## 族目录声明义务（.mdp_family——会话 SDK 契约面）

建模会话的族目录 = 运行期 discover() 的 model 词表过滤带 `.mdp_family` 声明者——
**强制完整声明：未声明 = 对会话不可见**（DAG 词表不受影响）。第三方模型作者在节点
函数上声明即入目录（第二种扩展形态的对偶面：加节点 vs 加整个环境）：

```python
MY_MODEL.mdp_family = {
    "structure": {"cat_support": "encode", "scaling_need": True, "min_samples": 30},  # 相容性自声明
    "regularizers": ("my_l2",),  # 正则面（可选——出题规则的正则收窄/pin 目标；无正则旋钮可缺省）
    "machinery": make_machinery(MY_MODEL, _model_cls),  # 会话直调的 estimator 机器
    # （task, x_preprocessors, seed=None, **overrides) -> (make_pipe, base_params, param_space)
}
```

`.mdp_family` 是节点函数上的**唯一尾巴**（structure 直书内嵌、正则面并入——不再
散挂 `.structure`/`.regularizers` 裸属性）。

签名即搜索空间的单一真源（默认 `ContinuousSpace`/`DiscreteSpace` = 动态维、常量默认 =
静态固定）——`.mdp_family` 不复制空间（防第二真源漂移）；`machinery` 经工厂从签名
派生（不手写第二份缺省表）。**声明在 `registry.families()` 聚合期过准入校验**（缺键/
值域错/未知键 → raise 并附本节指引——错误信息与文档同源）；破坏性变更受 CHANGELOG
语义化版本约束。

## 会话观察面（#35/#64）

- `diagnose.py`：OOF 观察面诊断算子（error_concentrated / slice_collapse / fold_std /
  complexity_edge + 体征面 underfit_high_close / underfit_loss_plateau /
  overfit_train_val_gap / overfit_val_loss_rise——train 折分与 θ* 曲线探针证据，
  设计稿 2026-10-28）——run_hpo 后自动接线（findings 落账本 + 响应浮出），零步数
- `report.py`：OOF 图集（挂 run 下，run_id 寻址）/ 泰勒聚合（终训节点副产物
  fan-in——`taylor_byproducts` 边函数断言各模型 y_test 逐位同一）/ 报告素材
  （`modeling_report_skeleton`：表格从账本渲染 + 图片清单 + 叙事槽）；终局报告不在
  引擎——π 撰写后 `finish(pdf)` 收编进 deliverables（tmp/rename 原子落盘，不编译不验内容）
- `propose_rules.py`：出题规则实例（FamilyRule/FERule 注册在 modelRuleEngine 默认实例）——
  findings 驱动层经 `rule_engine_rwa` → fire/poll 进菜单（会话实例 pkl 持载自定义规则）
