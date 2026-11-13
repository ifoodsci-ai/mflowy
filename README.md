# MFlowy — 实验科学闭环的 Agent 能力底座（PILOT）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-Server-7B61FF.svg)](packages/mcp/mflowy/mcp/server.py)

MFlowy 是 **MCP-native** 的 AI4S 能力层，实现 **PILOT 闭环**（**P**rior-**I**nherited **L**oop for **O**ptimization and **T**esting，先验继承闭环优化测试）：从个人历史实验数据出发——数据清洗、模型预训练、贝叶斯优化推荐下一轮实验——湿实验结果回流，学习门禁判定继续还是带着失败信号回炉重训。每个子系统都是一场 agent 会话，统一范式：**先验讨论建档（剧本问卷）→ 核心执行（POMDP 会话）→ 产出与先验更新**。系统设计全文见 [docs/pilot-loop.md](docs/pilot-loop.md)。

```mermaid
flowchart LR
    D["历史实验数据"] --> S["Clean 子系统<br/>survey 剧本 → 清洗 POMDP"]
    S -->|final_path| M["Model 子系统<br/>model-brief → 建模 POMDP"]
    M -->|模型 loader| B["BO 子系统<br/>Virtual 筛选 + 推荐台账"]
    B -->|推荐 CSV| W["Wet-lab（外部）<br/>人工/自动化实验室"]
    W -->|结果回流 + 学习门禁| S
```

| 子系统 | 先验建档 | 核心执行 | 产出 | 状态 |
| ------ | -------- | -------- | ---- | ---- |
| **Clean** | survey 剧本 → DomainCard | 清洗 POMDP 会话（`clean_*` 六操作） | 终态数据 + 清洗计划 + 决策日志 | ✅ 已实现 |
| **Model** | model-brief 剧本（+research-review 文献候选）→ CVCard | 建模 POMDP 会话（`modeling_*`）+ compute 工具 | 模型 loader + findings + 报告 | ✅ 已实现 |
| **BO** | bo-brief 剧本 → BO 先验卡 | 推荐会话 + Virtual 筛选 | 推荐台账 CSV + 学习门禁信号 | 🚧 设计中（[决策图 #171](https://github.com/ifoodsci-ai/mflowy/issues/171)） |
| **Wet-lab** | 外部环境（无设备 API，人肉执行） | — | 结果台账回流 | ⬜ 外部 |

方法论叙事见 [docs/research-flow.md](docs/research-flow.md)（前两程的一局：五律总纲 + 五幕案例）；子系统决策点的研究依据见 [docs/method/](docs/method/)（清洗侧 / 建模侧，零词表绑定）。

## Agent 自主回路

同一套工具目录（`tools/list` = 已安装能力），两种驱动面：

- **会话面**（研究流主形态）：人点火剧本（`.agents/skills/*`）× MCP 会话工具（start / observe / apply / rollback / finish / abandon）× agent 决策——POMDP 的环境面在引擎、策略面在对话中的 agent；预算、版本树、决策日志、审计由会话壳承载。
- **图面**（单能力 / 整图编排，YAML 工作流）：

```text
能力目录（@node 节点 / @edge 边函数自注册） ──组装──▶ YAML 图（flow 工件） ──执行──▶ 实验记录（DAGResult + MLflow）
```

| 阶段 | 工具 | agent 用它做什么 |
| ---- | ---- | --------------- |
| 发现 | `list-nodes` / `get-node-info` | 枚举 `namespace.node_name` 能力目录（entry points 插件注册，签名式富化显依赖）；参数契约由函数签名内省进 inputSchema，拿到即会用 |
| 组装 | `validate_modeling_steps` | 预检 LLM 生成的 YAML 图——节点名、图结构错误在编译期抓出（结构化错误详情供 agent 自我纠正，另出带标签数据边的 mermaid 预览），通过后再执行 |
| 执行 | `modeling` / `data_profile` / `eda` / `predict` … | 运行单能力或整图；结构化 `DAGResult` 逐节点回报 run_id/状态/输出 |
| 读回 | `get_experiment` / `get_run` | 查 MLflow 实验仪表盘（本体+run 目录）与 run 详情（params/metrics/tags/产物目录），据实决定下一步 |

回路可靠运转的两块基石：可枚举的插件词表（`namespace.node_name` 节点 + `edges` 边函数，内置 entry points + 第三方 `mflowy.plugins` 组）让 LLM 与人共享同一图语言，YAML 可序列化往返、生成图可被工具改写复用；复用旧结果是显式 run_id 引用、不做静默缓存——agent 的控制流永远可预测。


## 组件架构（C4）

内核冻结、生态外挂：全部能力以插件存在（内置能力也在独立包、与第三方同构），身份是 entry point 声明（`namespace.node_name`），安装即注册——uv/pip 装上即进入目录，无注册中心、无本地状态文件。

```mermaid
C4Container
title MFlowy 组件架构（C4 Container）

Person(agent, "自主 Agent", "数据分析 / 建模 / 预测 agent（Claude Code、Cursor 等）")
Person(dev, "能力作者", "写 @node 插件的开发者")

System_Boundary(sys, "MFlowy —— uv workspace · PEP 420 namespace · 六 distribution 锁步") {
    Container(mcp, "mflowy-mcp", "MCP 层", "工具注册面（平台工具 + 域工具发现）· mflowy-mcp stdio server · mflowy JSON runner · 平台登录 · 遥测")
    Container(data, "mflowy-data", "数据域", "清洗 POMDP 动作域：load/statistic/detect/clean + plot(data_analysis)；数据栈硬依赖，装包即得完整数据域")
    Container(model, "mflowy-model", "建模域", "建模 POMDP 动作域：x_y/cv/x_transformer/model + plot(evaluation/interpretability) · compute 工具与 JobProvider 执行策略；本地建模重栈（CPU 六件 + torch）住 [local] extra，瘦装 = 纯远程 server")
    Container(driver, "mflowy-driver", "DAG 内核", "Builder YAML 增量编译（平铺叙事·编译即校验）· Context 血缘 DAG · LIFO 拓扑调度 + 失败血缘剪枝 · discover 插件目录（零 import 查询/惰性加载）· @node/@edge 插件 SDK + mlflow 默认尾链")
    Container(utils, "mflowy-utils", "共享工具", "mlflow 封装 · jinja 沙箱 · 日志与 stdout 捕获 · 文件指纹 · 脚本安全门")
}

System_Boundary(eco, "第三方生态") {
    Container(tpkg, "mflowy.plugins 插件包", "第三方插件", "自有 pyproject 声明 entry points（namespace.node_name），随 uv --with / 镜像定制安装")
}

System(mlflowsys, "MLflow Tracking", "实验记录：runs · 参数 · 指标 · 模型与产物 · 血缘 tag")
System(remote, "远程执行环境", "K8s Job · 集群 · Serverless（自定义 JobProvider 实现接管）")

Rel(agent, mcp, "MCP stdio（mflowy-mcp）/ subprocess（mflowy）", "tools/list · tools/call")
Rel(dev, tpkg, "pyfunc + @node + entry point 声明")
Rel(mcp, data, "依赖：编排数据域工具")
Rel(mcp, model, "依赖：编排建模域工具")
Rel(data, driver, "依赖：@node/@edge SDK · 遥测契约")
Rel(model, driver, "依赖：@node/@edge SDK · 遥测契约")
Rel(driver, utils, "依赖")
Rel(tpkg, driver, "base 依赖：SDK；安装即注册进 discover 目录（可覆盖内置同名项）")
Rel(driver, mlflowsys, "每节点一个 run · 血缘 tag · LIFO 调度记录")
Rel(data, mlflowsys, "log_* 观测：图表 / 表格 / 领域日志落盘")
Rel(model, mlflowsys, "log_* 观测：图表 / 表格 / 领域日志落盘")
Rel(model, remote, "compute 工具执行委派", "JobProvider 协议")
```

| distribution | 角色 | 职责 |
| ------------ | ---- | ---- |
| `mflowy` | 聚合包（PyPI 入口） | extras = 形态选择器：`[mcp]` / `[local]`——域包直装（`mflowy-data` / `mflowy-model`），`mflowy[mcp]` 起 server（纯远程瘦装）、`mflowy[mcp,local]` = server + 本地建模重栈；聚合 `[data]`/`[model]`/`[deep]` 已退役 |
| `mflowy-mcp` | 用户面 | 工具注册面（平台工具 + 域工具发现）与三种入口、平台登录、遥测与同意门 |
| `mflowy-data` | 数据域 | 清洗 POMDP 动作域（load/statistic/detect/clean + EDA 图）；数据栈硬依赖 |
| `mflowy-model` | 建模域 | 建模 POMDP 动作域（x_y/cv/x_transformer/model + 评估/解释图）+ compute 工具执行策略（JobProvider）；本地建模重栈（sklearn/lightgbm/xgboost/catboost/shap/optuna/torch）住 `[local]` extra；两域包同为第三方插件的活参考（抄走 pyproject + hook，改 entry points 组即成插件包） |
| `mflowy-driver` | 内核 | 编译与调度，不实现任何业务能力；插件 SDK 与词表校验所在 |
| `mflowy-utils` | 底座 | 无业务语义的共享工具层 |

依赖方向单向（utils ← driver ← data ← model ← mcp），机器可检查（`tests/test_workspace.py` 断言锁步版本、namespace 铁律与依赖方向）。三入口共享同一套工具实现：

| 入口 | 命令 | 场景 |
| ---- | ---- | ---- |
| **CLI 通道** | `mflowy <tool> '<json args>'` | 命令行、K8s Job 容器、subprocess；裸装即用（零 SDK） |
| **MCP 通道** | `mflowy-mcp` | stdio server，MCP 客户端 / agent 接入；SDK+遥测住 `[mcp]` extra，本地建模组合 `mflowy[mcp,local]`（纯 `mflowy[mcp]` = 远程瘦装 server） |
| 直接 import | `mflowy.{mcp,data,model}.tools` pyfunc | 宿主程序内嵌调用 |

> CLI（`mflowy`）是 MCP 工具层的命令行通道，与 MCP server 共享同一套工具实现与 JobProvider 委派，不是独立架构。

各组件深读：[driver — 内核架构与设计哲学](packages/driver/README.md) · [data — 数据域 catalog](packages/builtin_plugins/data/README.md) · [model — 建模域 catalog / 远程执行（JobProvider）](packages/builtin_plugins/model/README.md) · [mcp — 工具 / 平台登录 / 遥测](packages/mcp/README.md) · [utils — 共享工具](packages/utils/README.md)

## 快速开始

两种通道、双命令入口：`mflowy <tool> '<json>'` = CLI（runner）；`mflowy-mcp` = MCP server（需 `[mcp]`）。

### 方式一：PyPI

> ⚠️ **PyPI 当前为 0.1.2（2026-08-23，Alpha）——参数边 / @edge BREAKING 重构之前的旧架构**，与本仓库文档描述的能力集不一致。0.2.0 发布前请用方式二（源码）安装；届时本节恢复推荐位。

```bash
# MCP server (stdio) — 完全体（数据分析 + 本地建模重栈）；国内镜像参数可省（网络可达 PyPI 时）
uvx --index-strategy unsafe-best-match \
    --default-index https://mirrors.aliyun.com/pypi/simple/ \
    --index https://download.pytorch.org/whl/cpu \
    --from "mflowy[mcp,local]" \
    mflowy-mcp

# 纯远程瘦装 server（零建模重栈；计算走 remote provider，需平台登录）
uvx --from "mflowy[mcp]" mflowy-mcp

# 轻量分析（数据域随 [mcp] 随行，无建模重栈，无需 CPU 索引参数）
uvx --from "mflowy[mcp]" mflowy data_profile '{"file_path": "..."}'

# 挂载第三方插件包（安装即注册）
uvx --from "mflowy[mcp,local]" --with mflowy-extra mflowy-mcp

# 或常规安装（pip / uv pip）
pip install "mflowy[mcp,local]"
```

MCP 客户端配置（stdio）——[`.mcp.json.example`](.mcp.json.example) 为模板：

```json
{
  "mcpServers": {
    "mflowy": {
      "type": "stdio",
      "command": "uvx",
      "args": [
        "--index-strategy", "unsafe-best-match",  // torch CPU 索引必需（见下方「启动说明」）
        "--default-index", "https://mirrors.aliyun.com/pypi/simple/",  // 可选：镜像
        "--index", "https://download.pytorch.org/whl/cpu",  // [local] 需要（torch 真合并随行）
        "--from", "mflowy[mcp,local]",
        "mflowy-mcp"
      ]
    }
  }
}
```

### 方式二：源码（**0.2.0 发布前推荐**；开发 / 贡献）

```bash
git clone https://github.com/ifoodsci-ai/mflowy.git && cd mflowy
uv sync --all-packages --all-extras --all-groups

uv run mflowy list-nodes                                         # 查看支持的节点目录（base，无数据栈，签名式显依赖）
uv run mflowy data_profile '{"file_path": "..."}'  # 数据分析工具
uv run mflowy-mcp                                             # MCP server（stdio）
```

更多开发命令（测试 / lint / 构建）见 [AGENTS.md](AGENTS.md) 或 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 环境变量

| 变量                    | 用途                                                                                           | 示例                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `MLFLOW_TRACKING_URI` | Tracking server URI（未设置时 workflow 与查询工具同落固定库 `~/.mflowy/mlflow.db`） | `postgresql://user:pwd@host:5432/mlflow` |
| `MFLOWY_JOB_PROVIDER` | JobProvider 解析：`local`（默认）或 `module:Class`（自定义实现）                           | `my_pkg.job_provider:MyRemoteProvider`   |
| `PYTHONPATH`          | 自定义 JobProvider 模块的包根                                                                  | `/srv/my-provider`                       |
| `MFLOWY_TELEMETRY`   | 遥测模式：`ask`（默认，首次工具调用时询问）/ `on` / `off`（显式设置覆盖 settings.json，见下方「遥测」） | `on`                                  |

### 启动说明

- **入口名 `mflowy-mcp`**：刻意避开 mcp SDK 自带的同名 `mcp` CLI（`mcp.cli:app`）——uvx 解析 `mcp` 命令时可能命中 SDK 侧导致启动失败
- **extras 内联在 `--from` spec**：uvx 的 `--extra` 需新版 uv，内联写法兼容性最好
- **离线分发用 wheels**：`make build-whl` 产出六 wheel（`dist/*.whl`，六 distribution 锁步），`uv pip install --find-links <dist目录> "mflowy[mcp,local]"` 安装（K8s 镜像构建 / 内网场景；Dockerfile 即此形态，支持 `--build-arg MFLOWY_EXTRA_MODULES=...` 定制插件）
- **torch CPU 索引**（`--index` pytorch-cpu + `--index-strategy unsafe-best-match`）为 [local] 必需（`mflowy[local]` / `mflowy[mcp,local]`——torch 随 [local] 真合并随行；纯 `mflowy[mcp]` 瘦装不装 torch 无需索引）：uvx 不读 pyproject 的 `[tool.uv.sources]`，缺省时 torch 解析为 CUDA 全家桶（2–3GB）；`unsafe-best-match` 须与 pytorch 索引同用，否则 first-index 策略会因该索引上的旧版 requests 解析失败

## 遥测（Telemetry）

MCP 工具调用诊断采集，同意制、默认 `ask`，端点不可达时完全透明不影响工具调用，仅覆盖 MCP 入口。隐私契约见 [PRIVACY.md](PRIVACY.md)，接入与配置详情见 [packages/mcp/README.md](packages/mcp/README.md)。

## 贡献

欢迎任何形式的贡献（功能、修复、文档、案例）。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)（开发流程与约定）、[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)、[PRIVACY.md](PRIVACY.md)（遥测隐私契约）与 [SECURITY.md](SECURITY.md)（漏洞披露）。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 文档

- **系统与叙事**：[PILOT 系统设计](docs/pilot-loop.md)（三子系统闭环）· [研究流方法论](docs/research-flow.md)（前两程案例叙事，agent 可直接采用的工作规约）· [路线图](docs/roadmap.md)
- **子系统方法论**：[数据清洗](docs/method/data-cleaning.md) · [模型训练](docs/method/model-training.md)（研究领域知识）· [数据域 catalog](packages/builtin_plugins/data/README.md) · [建模域 catalog / 远程执行](packages/builtin_plugins/model/README.md)
- **引擎与工程**：[架构与贡献指南](AGENTS.md) · [元词汇词表](CONTEXT.md)（引擎内核 + 三子系统） · [DAG 内核](packages/driver/README.md) · [MCP 层：工具 / 平台登录 / 遥测](packages/mcp/README.md) · [共享工具层](packages/utils/README.md)
