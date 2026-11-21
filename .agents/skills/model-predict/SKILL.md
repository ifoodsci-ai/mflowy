---
name: model-predict
description: 预测应用手册(mflowy)。已训模型对新数据出预测值——用户要求"预测/打分/批量推断"且模型已训练完成（run_id 在手）时使用。predict 工具含 loader/run_id 寻址语义（model = 训练 run 的 id，数据文件直传）。应用面在建模 POMDP 之外（#233 拆分）；bo-suggest 回填期的 predict 补点值为内联用法，不经本手册。
disable-model-invocation: true
---

# Model-Predict Skill：预测应用（predict 工具手册）

建模 POMDP 之外的应用面：拿训练交付的模型引用（run_id）对新数据出预测值。
你的职责：备数据、寻址 run、调工具、回读结果——无会话、无回合循环。

## 用法

```
predict(data=<新数据绝对路径>, model=<run_id>)
```

- **data**：csv/xlsx/parquet 文件绝对路径，或 py 文件 `'path:func'` 引用（数据加载
  函数——复杂预处理在函数里做）；
- **model（run_id 寻址语义）**：训练 run 的 id——从训练输出 RunInfo 或
  `get_experiment(experiment_id)` 获取；modeling 主剧本 finish 交付的 loader 引用
  即此 run_id（绝对路径直传，下游按它寻址）。

## 受理面（JobOutcome）

- local 模式：完成等待（finished 内嵌 DAGResult——预测表即结果）；
- remote 模式：受理即返（submitted + job_id）——先 `ensure_platform_experiment(...)`
  就绪实验；在途观察走 `get_platform_job_logs(job_id)` 轮询（terminal=true 即停）；
  产物经 `download_platform_job_artifacts(job_id)` 拉取（outputs 下 {job_id}.zip）。

## 纪律

- 预测数据须与训练特征面一致（列名/顺序/单位——承袭清洗词表）；不一致即报错
  交用户裁定，不静默改数据；
- 分类任务按需说明输出列语义（预测标签 vs 概率）；
- 预测值未经实验验证时如实标注（bo-suggest 台账尾部提醒同源）。

## 参考

- 工具发现：`mflowy predict -h` 看全参与用法（参数面以 help 为单源）；
- 姊妹技能：model-explain（SHAP 解释）；模型交付来源：modeling（finish loader 引用）。
