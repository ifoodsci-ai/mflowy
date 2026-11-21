---
name: model-explain
description: 模型解释应用手册(mflowy)。训练完成后解释特征贡献与依赖关系——用户要求"解释模型/特征重要性/SHAP/特征依赖"且模型已训练完成（run_id 在手）时使用。explanation 工具（SHAP 分析：modeling_steps YAML + run_id 寻址）。应用面在建模 POMDP 之外（#233 拆分）。
disable-model-invocation: true
---

# Model-Explain Skill：模型解释应用（explanation 工具手册）

建模 POMDP 之外的应用面：对已训模型做 SHAP 解释性分析（特征贡献 + 依赖关系）。
你的职责：组 modeling_steps、寻址 run、调工具、回读图件——无会话、无回合循环。

## 用法

```
explanation(modeling_steps=<建模步骤 YAML 明文>, model=<run_id>, name=, desc=, lowess_frac=)
```

- **modeling_steps**：与 modeling 工具相同的 YAML 明文（顶层步骤列表片段：
  load→clean→x_y→cv→transformer→model；`- name:` 开头、无 name/description/flow
  外壳）——解释对象 = 该图挂接的已训模型；模板可从 `mflowy modeling -h` 揭示的
  用法起步；
- **model（run_id 寻址语义）**：训练 run 的 id——从训练输出 RunInfo 或
  `get_experiment(experiment_id)` 获取；
- **lowess_frac**：依赖图 LOWESS 平滑比例（缺省 0.3）。

## 受理面（JobOutcome）

- local 模式：完成等待（finished 内嵌 DAGResult——SHAP 汇总/依赖图即结果）；
- remote 模式：受理即返（submitted + job_id）——先 `ensure_platform_experiment(...)`
  就绪实验；在途观察走 `get_platform_job_logs(job_id)` 轮询（terminal=true 即停）；
  产物经 `download_platform_job_artifacts(job_id)` 拉取（outputs 下 {job_id}.zip）。

## 纪律

- 解释结论须引用图件证据（特征名/方向/量级），不凭印象叙述；
- SHAP 图件是过程工件——入报告/交用户时落明确路径。

## 参考

- 工具发现：`mflowy explanation -h` 看全参与用法（参数面以 help 为单源）；
- 姊妹技能：model-predict（预测应用）；模型交付来源：modeling（finish loader 引用）。
