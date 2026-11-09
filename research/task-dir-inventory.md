# 远程 task_dir 产物清单与 zip 内容边界（#155）

调研方式：代码阅读（read-only）+ 镜像 ENV 真跑实证。实证环境：
`MFLOWY_TASK_DIR=<tmp>/ws2`、`MLFLOW_TRACKING_URI=sqlite:///<tmp>/ws2/mlflow.db`（四斜杠）、
`HOME=<隔离目录>`、经 `mflowy <tool> '<json>'` runner 入口 + LocalJobProvider，iris（分类）/一维回归（回归）小数据集，
覆盖五种 compute 工具全部行为分支（search / training / explanation / predict / inverse_optimization，另补 MLP 与 study_storage_path）。
版本：mlflow 3.15.2、optuna 4.9.0。下表未标注「推断」的均为实证结果。

## 一、五工具 × task_dir 产物清单

### 共同底座（每工具必有）

| 路径 | 产生者 | 是否必进 zip | 典型体积（实证） |
| --- | --- | --- | --- |
| `mlflow.db` | `MLFLOW_TRACKING_URI` 指向的 sqlite（experiments/runs/tags/metrics/params/logged_models 表；mlflow 3.x 全 schema 60+ 表） | **必须**（`get_experiment`/`get_run` 查询面唯一真源） | 空库即 ~870KB；iris 级 7 个 workflow 后 920KB |
| `.mlruns/<run_id>/artifacts/**` | 每节点一个 mlflow run 的工件（experiment 建立时 `artifact_location=file://<task_dir>/.mlruns`） | **必须** | 见分表 |
| `.mlruns/models/m-<uuid>/artifacts/**` | `log_model(pipe, name="final")`——mlflow 3.x logged-model 本体：`model.pkl`、`MLmodel`、`conda.yaml`、`requirements.txt`、`python_env.yaml`、`input_example.json`、`serving_input_example.json`；predict/explanation 的 loader 触发时另有 `registered_model_meta` | **必须**（模型本体不随 run 工件目录走，缺它则 loader/`download_run_artifact` 失效） | 34–38KB/模型（sklearn RF）；随模型规模增长 |

注：**没有** `mlflow.db-wal`/`-shm` 残留（mlflow 正常关连接；若容器被 SIGKILL 理论上可残留，zip 一并打包即可，勿排除）。

### 1. hyperparameters_search（search 行为：model 步参数未全 pin → inner CV 选参）

| 路径 | 产生者 | 是否必进 zip | 典型体积 |
| --- | --- | --- | --- |
| `mlflow.db` 中 search run + `trial_N` 子 run 的 params/metrics/tags | `search()` 每 trial 记 run | 必须（θ*/u_inner 证据就在 db 里） | 行级，不占地 |
| `.mlruns/<cv.run>/artifacts/split.json`、`split.pkl` | `log_cv` 中间件 | 必须 | ~0.6KB + ~10KB |
| `study_<name>.db`（仅当显式传 `study_storage_path`） | optuna `sqlite:///<该路径>` + `load_if_exists`（增量续跑史） | 必须（否则续跑史丢失）；**路径必须给 task_dir 下绝对路径**——相对路径按 sqlite 语义落进程 CWD，不进 zip | 112KB（2 trial） |
| （工件文件） | 无——search 行为零 artifact 文件（实证） | — | — |

### 2. modeling（training 行为：model 步参数全 pin）

在共同底座之上：

| 路径 | 产生者 | 是否必进 zip | 典型体积（实证） |
| --- | --- | --- | --- |
| `.mlruns/<model.run>/artifacts/X_test.parquet`、`y_test.parquet`、`y_pred.parquet` | `#66` holdout 副产物（`_pipeline.py`） | **必须**——explanation 的 SHAP 测试面就取它（`download_run_artifact(model, "X_test.parquet")`） | 1.2–3.6KB（iris 级） |
| `actual_vs_predictions.parquet`、`test_probabilities.parquet`（分类）、`classification_report.json`（分类） | 终评可视化 | 建议 | ≤4KB |
| 评估图 png（300dpi）：分类 confusion/roc/pr/calibration ~170–270KB/张；回归 prediction_scatter/residual_scatter/bland_altman/error_distribution ~170–700KB/张 | `_evaluation_plots` | 建议（查询工具展示面） | **png 是体积大头** |
| `.mlruns/models/m-<uuid>/artifacts/**`（终训模型本体） | `log_model(name="final")` | **必须** | 34KB+ |
| `lightning_logs/lightning_logs/version_N/{metrics.csv, hparams.yaml, checkpoints/epoch=N-step=M.ckpt}`（仅 MLP，[deep] extra） | pytorch_lightning `default_root_dir=task_dir()/lightning_logs`（注意**双层目录**：锚目录 + lightning 自加 `lightning_logs/version_N`）；无显式 ModelCheckpoint → 缺省每 version 存一个 last ckpt，**每个 trial 一个 version** | 可选——模型本体已在 `.mlruns/models/`，诊断/续调才需要；体积风险源 | 实测 20KB（16-8 微网）；随网络规模 ×HPO trial 数线性放大 |
| 多模型时的 `plot__*` 对比图 run 工件（prediction_scatter/confusion_matrix/taylor/bland_altman/roc/pr/calibration） | MODELING_TEMPLATE `multi_model` 分支 | 建议 | 同评估图量级 |

### 3. explanation（SHAP）

| 路径 | 产生者 | 是否必进 zip | 典型体积（实证） |
| --- | --- | --- | --- |
| `.mlruns/<shap.run>/artifacts/shap_values.parquet` | `log_shap_explanation` | 必须 | 12.8KB |
| `.mlruns/<plot.run>/artifacts/shap_summary_N.png`、`shap_dependence_N.png` + 各图配套 `_class_K.parquet` | `plot__shap_summary`（top_n=10）/`plot__shap_dependence` | 建议 | **summary ~510KB/张、dependence ~1.0–1.1MB/张，多分类每类一套**（三分类 4 图 = 4.2MB）——单工具最大体积项 |
| `.mlruns/<cv.run>/artifacts/split.json`、`split.pkl` | 剪枝 steps 重跑 load/x_y/cv.split（新 run） | 建议 | 同上 |
| `models/m-.../registered_model_meta` | loader 读模型时的登记元数据 | 随 models/ 必须 | 37B |
| （前置条件）model run 的 `X_test.parquet` 在**同一** mlflow.db | `run_artifacts(model)` fail-loud 检查 | ⇒ 跨任务引用必须同库（见风险 2） | — |

### 4. predict

| 路径 | 产生者 | 是否必进 zip | 典型体积 |
| --- | --- | --- | --- |
| `.mlruns/<predict.run>/artifacts/prediction.parquet` | `log_prediction` | 必须 | 1.2KB |
| `models/m-.../registered_model_meta` | loader | 随 models/ | 37B |

### 5. inverse_optimization

| 路径 | 产生者 | 是否必进 zip | 典型体积 |
| --- | --- | --- | --- |
| `.mlruns/<search_input.run>/artifacts/search_results.parquet` | `log_search_input` | 必须 | 2.3KB（50 trial） |
| `models/m-.../registered_model_meta` | loader | 随 models/ | 37B |

### 其他边界事实

- **输入数据文件**：workspace（=task_dir）里上传的 csv/parquet 会随 zip 打包（实证目录中 iris.csv 等 4KB×3）。复现审计友好，但同一数据重复任务会重复占配额。
- **失败任务**：DAG 启动即 `ensure_experiment`——experiment 行已写 mlflow.db，工具级失败也留半成品库；无工件文件。
- **Default 实验（id 0）**：mlflow 建库时自带 `artifact_location=<CWD>/mlruns/0`（非 file:// 形态）。mflowy 全部走具名实验 + 显式 `artifact_location=file://<task_dir>/.mlruns`，工件从不落 CWD（实证 CWD 零残留；`mlflow.py` 落盘壳的 active-run 守卫也拦住了孤儿 run 面）。

## 二、artifact_uri 形态（实证，sqlite 直查）

```
experiments.artifact_location        = file:///abs/task_dir/.mlruns
runs.artifact_uri                    = file:///abs/task_dir/.mlruns/<run_id>/artifacts
logged_models.artifact_location      = file:///abs/task_dir/.mlruns/models/m-<uuid>/artifacts   ← mlflow 3.x 第二种形态
（experiments 第 0 行 Default        = /abs/cwd/mlruns/0   ← 仅为行，从未写入）
```

**结论**：`artifact_root_remote=/user-data/workspace/.mlruns → artifact_root_local=<local_task_dir>/.mlruns` 的前缀映射成立，但映射键必须取**实验级** `.mlruns` 根（不能取 run 级 artifacts 路径），否则 `models/m-<uuid>/` 形态的模型工件路径转换不了。

## 三、HOME 侧写入（实证：隔离 HOME 真跑五工具）

| 路径 | 触发者 | 说明 |
| --- | --- | --- |
| `$HOME/.mflowy/`（空目录） | `mlflow.py::setup()` 中 `os.getenv("MLFLOW_TRACKING_URI", default_tracking_uri())`——**默认值无条件求值**，`default_tracking_uri()` 里 `mkdir(exist_ok=True)` 即使 env 已设也执行 | 仅空目录；env 已设则不写库文件。代码位 `packages/utils/mflowy/utils/mlflow.py:266-283` |
| `$HOME/.config/mlflow/telemetry.json` | mlflow 自身安装指纹（非 mflowy 遥测） | 125B |
| `$HOME/.cache/matplotlib/fontlist-*.json`、`$HOME/.config/matplotlib/` | matplotlib 首次出图构建字体缓存 | ~30KB |
| ~~settings.json / api.key~~ | mflowy 遥测只在 MCP server 挂线（`telemetry.wire_agentcat` 在 `server.py:73`），JSON runner 路径不触发 | 实证隔离 HOME 无此二文件 |
| ~~credentials.json / *_sessions~~ | login / POMDP 会话工具路径，五个 compute 工具不触碰 | — |

全部在 HOME、不在 task_dir → **不进 zip**；per-pod 一次性、无泄漏面。真实 `~/.mflowy` 对比快照跑前跑后零变化（所有实证 run 都用隔离 HOME）。

## 四、zip include / exclude 建议（1024Mi emptyDir 配额）

**include（必含）**：`mlflow.db`、`.mlruns/` 整树（run 工件 + `models/` 模型本体）、`study_*.db`（若用户传了 study_storage_path）、输入数据文件（小，审计友好）。

**exclude 候选（按体积风险排序）**：
1. `lightning_logs/`——模型本体已在 `.mlruns/models/`，查询与复跑都不需要它；MLP 大网 ckpt × HPO trial 数是最可能顶爆配额的项。
2. 输入数据（若上传侧已有持久备份，可省）。

**体积实证**：iris 级 8 个 workflow（含 SHAP 三分类全套图 + 3 个终训模型 + MLP）workspace 原始 ~9.9MB，zip 后 **6.72MB**——`.mlruns` 8.9MB 里 png 占绝对大头。1024Mi 配额 ≈ 该规模 ~150 次任务；但宽特征数据集的 shap_dependence（1MB/张 × top_n × 类别数）与大数据集 parquet 副产物会快速放大，建议 zip 前加 size guard（超阈值先剥离 lightning_logs 与 `_class_*.parquet` 图数据）。

**风险**：
1. **配额**：见上；zip 动作本身在 emptyDir 内进行时注意 zip 临时文件也占配额（建议流式写 `/user-data/outputs`）。
2. **跨任务模型引用**：explanation/predict/inverse 的 `model=<run_id>` 在同一 `MLFLOW_TRACKING_URI` 内寻址——per-pod emptyDir 意味着第二任务看不到第一任务的 run。需要 RemoteJobProvider 层把前一任务 zip 解回 workspace（或 workspace 复用）再跑，`X_test.parquet` 前置检查同理。
3. **相对 study_storage_path**：sqlite 相对路径按进程 CWD 解析 → 落 task_dir 外、不进 zip；provider 层应把该参数规范化为 task_dir 下绝对路径。
4. `mlflow.db` 体积下限 ~870KB（mlflow 3.x 全 schema），多小任务也只累积行级数据，不是配额风险。

## 五、实证命令与输出摘要

环境（每 run）：`HOME=<隔离> MFLOWY_TASK_DIR=<tmp>/ws2 MLFLOW_TRACKING_URI=sqlite:///<tmp>/ws2/mlflow.db MFLOWY_TELEMETRY=off`，入口 `/tmp/wf155-venv/bin/mflowy <tool> '<json>'`（worktree 外独立 venv，`UV_PROJECT_ENVIRONMENT=/tmp/wf155-venv uv sync --frozen --all-packages --all-extras --all-groups`）。

- `hyperparameters_search`（RF，n_trials=2，参数未 pin → search 行为）：新增 3 文件 = `mlflow.db`(872KB) + cv run `split.json`/`split.pkl`；search/trial run 零工件文件。
- `modeling`（RF 全 pin → training 行为，分类）：新增 20 文件 = split×2 + model run 12 工件（X_test/y_test/y_pred/actual_vs_predictions/test_probabilities/classification_report + 4 张评估 png）+ `models/m-<uuid>/` 7 文件（model.pkl 37.5KB 等）。
- `modeling`（RF 全 pin，回归）：17 文件，回归图组（prediction_scatter 401KB / error_distribution 697KB / bland_altman / residual_scatter）。
- `explanation`（model=<cls run_id>）：16 文件，shap_summary×3（510–520KB）+ shap_dependence×3（1.02–1.09MB）+ shap_values.parquet + 图数据 parquet + 重跑 split + registered_model_meta。
- `predict`（data=<5 行 csv>, model=<run_id>）：1 文件 `prediction.parquet`。
- `inverse_optimization`（direction={y: maximize}, n_trials=50）：2 文件 `search_results.parquet` + registered_model_meta。
- `modeling`（MLP，max_epochs=2）：`lightning_logs/lightning_logs/version_0/{metrics.csv,hparams.yaml,checkpoints/epoch=1-step=4.ckpt}`（12KB）。
- `hyperparameters_search`（study_storage_path=<task_dir>/study_RF_v1.db）：该路径 sqlite 落盘 112KB（2 trial）。
- 失败 run（如 explanation 引用无 X_test 的 search run）：experiment 行仍写入 mlflow.db，无工件。
- sqlite 直查（`runs`/`experiments`/`logged_models` 表）输出见第二节三行形态。
- 隔离 HOME 跑后：`.mflowy/`（空）、`.config/mlflow/telemetry.json`、`.cache|config/matplotlib/`；无 settings.json/api.key/credentials.json。CWD 目录全程为空。

主要代码出处：`packages/utils/mflowy/utils/path.py`（pin_task_dir）、`packages/utils/mflowy/utils/mlflow.py`（get_artifact_uri/log_* 家族/setup）、`packages/builtin_plugins/model/mflowy/model/job_provider/local.py`（_anchor_task_dir/study_storage）、`packages/mcp/mflowy/mcp/runner.py`（MFLOWY_TASK_DIR 钉住）、`packages/builtin_plugins/model/mflowy/model/model/_pipeline.py`（holdout 副产物/log_model）、`_mlp.py`（lightning 锚）、`packages/mcp/mflowy/mcp/telemetry.py`+`server.py`（遥测仅 server 挂线）。
