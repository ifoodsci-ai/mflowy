# MFlowy — 自主数据分析、训练与预测 Agent 的能力底座

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-Server-7B61FF.svg)](packages/mcp/mflowy/mcp/server.py)

MFlowy 是 **MCP-native** 的 ML 能力层：把数据分析、模型训练与预测打包成可枚举、可自描述的 MCP 工具目录，供自主 agent（或人）组装成可追溯的实验工作流。agent 不需要读文档——能力发现、图组装预检、执行与结果读回全部经工具完成。

```text
能力目录（@handler 节点 / @edge 边函数自注册） ──组装──▶ YAML 图（flow 工件） ──执行──▶ 实验记录（DAGResult + MLflow）
```

## Agent 自主回路

MFlowy 不内置 agent；它提供让 agent 自主运转的四个阶段，每个阶段都是已注册的 MCP 工具：

| 阶段 | 工具 | agent 用它做什么 |
| ---- | ---- | --------------- |
| 发现 | `list-nodes` / `get-node-info` | 枚举 `namespace.node_name` 能力目录（entry points 插件注册，签名式富化显依赖）；参数契约由函数签名内省进 inputSchema，拿到即会用 |
| 组装 | `validate_modeling_steps` | 预检 LLM 生成的 YAML 图——节点名、图结构错误在编译期抓出（结构化错误详情供 agent 自我纠正，另出带标签数据边的 mermaid 预览），通过后再执行 |
| 执行 | `modeling` / `data_profile` / `eda` / `predict` … | 运行单能力或整图；结构化 `DAGResult` 逐节点回报 run_id/状态/输出 |
| 读回 | `get_experiment` / `get_run` | 查 MLflow 实验仪表盘（本体+run 目录）与 run 详情（params/metrics/tags/产物目录），据实决定下一步 |

回路可靠运转的两块基石：可枚举的插件词表（`namespace.node_name` 节点 + `edges` 边函数，内置 entry points + 第三方 `mflowy.plugins` 组）让 LLM 与人共享同一图语言，YAML 可序列化往返、生成图可被工具改写复用；复用旧结果是显式 run_id 引用、不做静默缓存——agent 的控制流永远可预测。

研究方法论（划分先行 / 基线参照 / 单一变更等五律）见 [docs/research-flow.md](docs/research-flow.md)，可直接作为 agent 的工作规约。

## 组件架构（C4）

内核冻结、生态外挂：全部能力以插件存在（内置能力也在独立包、与第三方同构），身份是 entry point 声明（`namespace.node_name`），安装即注册——uv/pip 装上即进入目录，无注册中心、无本地状态文件。

```mermaid
C4Container
title MFlowy 组件架构（C4 Container）

Person(agent, "自主 Agent", "数据分析 / 建模 / 预测 agent（Claude Code、Cursor 等）")
Person(dev, "能力作者", "写 @handler 插件的开发者")

System_Boundary(sys, "MFlowy —— uv workspace · PEP 420 namespace · 五 distribution 锁步") {
    Container(mcp, "mflowy-mcp", "MCP 层", "全部工具定义（tools.py）· mflowy-mcp stdio server · mflowy JSON runner · JobProvider 执行策略 · 遥测")
    Container(data, "mflowy-data", "数据域", "清洗 MDP 动作域：load/statistic/detect/clean + plot(data_analysis)；数据栈硬依赖，装包即得完整数据域")
    Container(model, "mflowy-model", "建模域", "建模 MDP 动作域：x_y/cv/x_transformer/model + plot(evaluation/interpretability)；CPU 建模栈硬依赖，torch 栈隔离 [deep]")
    Container(driver, "mflowy-driver", "DAG 内核", "Builder YAML 增量编译（平铺叙事·编译即校验）· Context 血缘 DAG · LIFO 拓扑调度 + 失败血缘剪枝 · discover 插件目录（零 import 查询/惰性加载）· @handler/@edge 插件 SDK + mlflow 默认尾链")
    Container(utils, "mflowy-utils", "共享工具", "mlflow 封装 · jinja 沙箱 · 日志与 stdout 捕获 · 文件指纹 · 脚本安全门")
}

System_Boundary(eco, "第三方生态") {
    Container(tpkg, "mflowy.plugins 插件包", "第三方插件", "自有 pyproject 声明 entry points（namespace.node_name），随 uv --with / 镜像定制安装")
}

System(mlflowsys, "MLflow Tracking", "实验记录：runs · 参数 · 指标 · 模型与产物 · 血缘 tag")
System(remote, "远程执行环境", "K8s Job · 集群 · Serverless（自定义 JobProvider 实现接管）")

Rel(agent, mcp, "MCP stdio（mflowy-mcp）/ subprocess（mflowy）", "tools/list · tools/call")
Rel(dev, tpkg, "pyfunc + @handler + entry point 声明")
Rel(mcp, plugins, "依赖：工具编排内置能力")
Rel(plugins, driver, "依赖：@handler/@edge SDK · 遥测契约")
Rel(driver, utils, "依赖")
Rel(tpkg, driver, "base 依赖：SDK；安装即注册进 discover 目录（可覆盖内置同名项）")
Rel(driver, mlflowsys, "每节点一个 run · 血缘 tag · LIFO 调度记录")
Rel(plugins, mlflowsys, "log_* 观测：图表 / 表格 / 领域日志落盘")
Rel(mcp, remote, "建模类工具执行委派", "JobProvider 协议")
```

| distribution | 角色 | 职责 |
| ------------ | ---- | ---- |
| `mflowy` | 聚合包（PyPI 入口） | extras = 域选择器：`[data]` / `[model]` / `[mcp]` / `[deep]`——`mflowy[mcp]` 起 server、`mflowy[data|model]` 直接走 runner；`[stats]`/`[modeling]` 已退役 |
| `mflowy-mcp` | 用户面 | 全部 MCP 工具与三种入口、执行策略（JobProvider）、遥测与同意门 |
| `mflowy-data` | 数据域 | 清洗 MDP 动作域（load/statistic/detect/clean + EDA 图）；数据栈硬依赖 |
| `mflowy-model` | 建模域 | 建模 MDP 动作域（x_y/cv/x_transformer/model + 评估/解释图）；CPU 建模栈硬依赖，torch 栈住 `[deep]`；两域包同为第三方插件的活参考（抄走 pyproject + hook，改 entry points 组即成插件包） |
| `mflowy-driver` | 内核 | 编译与调度，不实现任何业务能力；插件 SDK 与词表校验所在 |
| `mflowy-utils` | 底座 | 无业务语义的共享工具层 |

依赖方向单向（utils ← driver ← data ← model ← mcp），机器可检查（`tests/test_workspace.py` 断言锁步版本、namespace 铁律与依赖方向）。三入口共享同一套工具实现：

| 入口 | 命令 | 场景 |
| ---- | ---- | ---- |
| **CLI 通道** | `mflowy <tool> '<json args>'` | 命令行、K8s Job 容器、subprocess；裸装即用（零 SDK） |
| **MCP 通道** | `mflowy-mcp` | stdio server，MCP 客户端 / agent 接入；SDK+遥测住 `[mcp]` extra，常与能力 extras 组合（如 `mflowy[mcp,model]`） |
| 直接 import | `mflowy.mcp.tools` pyfunc | 宿主程序内嵌调用 |

> CLI（`mflowy`）是 MCP 工具层的命令行通道，与 MCP server 共享同一套工具实现与 JobProvider 委派，不是独立架构。

各组件深读：[driver — 内核架构与设计哲学](packages/driver/README.md) · [data — 数据域 catalog](packages/builtin_plugins/data/README.md) · [model — 建模域 catalog](packages/builtin_plugins/model/README.md) · [mcp — 工具 / 远程执行 / 遥测](packages/mcp/README.md) · [utils — 共享工具](packages/utils/README.md)

## 快速开始

两种通道、双命令入口：`mflowy <tool> '<json>'` = CLI（runner）；`mflowy-mcp` = MCP server（需 `[mcp]`）。

### 方式一：PyPI

> ⚠️ **PyPI 当前为 0.1.2（2026-08-23，Alpha）——参数边 / @edge BREAKING 重构之前的旧架构**，与本仓库文档描述的能力集不一致。0.2.0 发布前请用方式二（源码）安装；届时本节恢复推荐位。

```bash
# MCP server (stdio) — 完全体（数据分析 + 建模）；国内镜像参数可省（网络可达 PyPI 时）
uvx --index-strategy unsafe-best-match \
    --default-index https://mirrors.aliyun.com/pypi/simple/ \
    --index https://download.pytorch.org/whl/cpu \
    --from "mflowy[mcp,model]" \
    mflowy-mcp

# 轻量分析（仅 [data]，无 torch，无需 CPU 索引参数）
uvx --from "mflowy[data]" mflowy data_profile '{"file_path": "..."}'

# 挂载第三方插件包（安装即注册）
uvx --from "mflowy[mcp,model]" --with mflowy-extra mflowy-mcp

# 或常规安装（pip / uv pip）
pip install "mflowy[mcp,model]"
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
        "--index", "https://download.pytorch.org/whl/cpu",  // [deep] 需要（仅 MLP）
        "--from", "mflowy[mcp,model]",
        "mflowy-mcp"
      ]
    }
  }
}
```

### 方式二：源码（**0.2.0 发布前推荐**；开发 / 贡献）

```bash
git clone https://github.com/ifoodsci-ai/mflowy.git && cd mflowy
uv sync --all-extras --all-groups

uv run mflowy list-nodes                                         # 查看支持的节点目录（base，无数据栈，签名式显依赖）
uv run --extra stats mflowy data_profile '{"file_path": "..."}'  # 数据分析工具
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
- **离线分发用 wheels**：`make build-whl` 产出六 wheel（`dist/*.whl`，六 distribution 锁步），`uv pip install --find-links <dist目录> "mflowy[model]"` 安装（K8s 镜像构建 / 内网场景；Dockerfile 即此形态，支持 `--build-arg MFLOWY_EXTRA_MODULES=...` 定制插件）
- **torch CPU 索引**（`--index` pytorch-cpu + `--index-strategy unsafe-best-match`）为 [deep] 必需（仅 MLP；`mflowy[model]` CPU 建模栈干净安装无需索引）：uvx 不读 pyproject 的 `[tool.uv.sources]`，缺省时 torch 解析为 CUDA 全家桶（2–3GB）；`unsafe-best-match` 须与 pytorch 索引同用，否则 first-index 策略会因该索引上的旧版 requests 解析失败

## 遥测（Telemetry）

MCP 工具调用诊断采集，同意制、默认 `ask`，端点不可达时完全透明不影响工具调用，仅覆盖 MCP 入口。隐私契约见 [PRIVACY.md](PRIVACY.md)，接入与配置详情见 [packages/mcp/README.md](packages/mcp/README.md)。

## 贡献

欢迎任何形式的贡献（功能、修复、文档、案例）。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)（开发流程与约定）、[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)、[PRIVACY.md](PRIVACY.md)（遥测隐私契约）与 [SECURITY.md](SECURITY.md)（漏洞披露）。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 文档

- [架构与贡献指南](AGENTS.md)
- [引擎元词汇词表](CONTEXT.md)
- [研究流方法论](docs/research-flow.md)（agent 可直接采用的工作规约）
- [路线图](docs/roadmap.md)
- [DAG 内核：设计哲学与架构](packages/driver/README.md)
- [数据域 catalog](packages/builtin_plugins/data/README.md) · [建模域 catalog](packages/builtin_plugins/model/README.md)
- [MCP 层：工具 / 远程执行 / 遥测](packages/mcp/README.md)
- [共享工具层](packages/utils/README.md)
