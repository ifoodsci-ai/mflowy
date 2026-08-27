# MFlowy — 自主数据分析、训练与预测 Agent 的能力底座

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-Server-7B61FF.svg)](packages/mcp/mflowy/mcp/server.py)

MFlowy 是 **MCP-native** 的 ML 能力层：把数据分析、模型训练与预测打包成可枚举、可自描述的 MCP 工具目录，供自主 agent（或人）组装成可追溯的实验工作流。agent 不需要读文档——能力发现、图组装预检、执行与结果读回全部经工具完成。

```text
能力目录（@handler 自注册） ──组装──▶ YAML 工作流（一等工件） ──执行──▶ 实验记录（WorkflowResult + MLflow）
```

## Agent 自主回路

MFlowy 不内置 agent；它提供让 agent 自主运转的四个阶段，每个阶段都是已注册的 MCP 工具：

| 阶段 | 工具 | agent 用它做什么 |
| ---- | ---- | --------------- |
| 发现 | `list_modules` / `get_module_info` | 枚举 `step.module` 能力目录（entry points 插件注册）；参数契约由函数签名内省进 inputSchema，拿到即会用 |
| 组装 | `validate_modeling_steps` | 预检 LLM 生成的 YAML DAG——模块名、图结构错误在编译期抓出，通过后再执行 |
| 执行 | `modeling` / `data_profile` / `eda` / `predict` … | 运行单能力或整图；结构化 `WorkflowResult` 逐节点回报 run_id/状态/输出 |
| 读回 | `list_runs` / `get_run` / `list_run_artifacts` | 查 MLflow 实验记录：对比历史 run、定位产物，据实决定下一步 |

回路可靠运转的两块基石：可枚举的 `step.module` 插件词表（内置 entry points + 第三方 `mflowy.plugins` 组）让 LLM 与人共享同一图语言，YAML 可序列化往返、生成图可被工具改写复用；复用旧结果是显式 run_id 引用、不做静默缓存——agent 的控制流永远可预测。

研究方法论（划分先行 / 基线参照 / 单一变更等五律）见 [docs/research-flow.md](docs/research-flow.md)，可直接作为 agent 的工作规约。

## 核心特性

- **MCP-native，三入口同源**：全部能力以 MCP 工具（pyfunc）暴露，MCP server（stdio）、JSON runner CLI（`cmd`）、直接 import（宿主内嵌）共享同一套实现与执行委派
- **能力目录，零手工注册**：能力 = 纯函数 + `@handler` 装饰器，构建期扫描生成 entry points 目录（`step.module` 即身份），新增能力只需一个 `.py` 文件；第三方包声明 `mflowy.plugins` entry points 即成插件；MCP schema 由函数签名内省自动生成
- **YAML 工作流是一等工件**：steps 串行 / branches 并行组成 DAG；可枚举的 step 词表让 LLM 与人共享同一图语言，图可序列化往返、跨工具改写复用
- **边按类型寻址**：节点按 step 检索最近上游而非点名引用——替换模块（如 XGB→LGBM）不需要改下游连线
- **实验即记录**：每次运行必留痕——结构化 `WorkflowResult`（逐节点 run_id/状态/输出）+ MLflow 全量追踪（参数/指标/模型/产物）+ 血缘 tag（`mflowy.input_steps`）；复用旧结果是显式 run_id 引用，不做静默缓存
- **中间件责任链**：数据注入、领域日志、mlflow 记录、错误即停在注册期一次性织入
- **执行环境可委派**：JobProvider 契约抽象 compute 工具的执行环境，内置本地实现，远程执行由自定义实现接入（见 [packages/mcp/README.md](packages/mcp/README.md)）

## 架构与调用方式

MFlowy 以 MCP 为架构主体：`packages/mcp/mflowy/mcp/tools.py` 定义全部插件工具，三种入口共享同一套工具。执行按工具类别分流：建模类经 JobProvider 委派执行（内置本地实现由 driver 内核编译调度 `builtin_plugins/` 能力，远程实现接管执行环境），分析类始终本地执行（详见 [packages/mcp/README.md](packages/mcp/README.md)）：

| 入口                | 命令                         | 场景                                     |
| ------------------- | ---------------------------- | ---------------------------------------- |
| MCP server（stdio） | `mcpSrv`                   | MCP 客户端 / agent（Claude Code、Cursor 等）接入 |
| JSON runner（CLI）  | `cmd <tool> '<json args>'` | 命令行、K8s Job 容器、subprocess         |
| 直接 import         | `mflowy.mcp.tools` pyfunc     | 宿主程序内嵌调用                         |

> CLI（`cmd`）是 MCP 工具层的命令行通道，与 MCP server 共享同一套工具实现与 JobProvider 委派，不是独立架构；历史独立 CLI（`mflowy run/validate/list-modules` 等）已废弃。

## 快速开始

两种运行形态，同一套入口（`mcpSrv` = MCP server / `cmd` = JSON runner）。

### 方式一：PyPI（推荐——无需克隆仓库）

```bash
# MCP server (stdio) — 完全体（数据分析 + 建模）；国内镜像参数可省（网络可达 PyPI 时）
uvx --index-strategy unsafe-best-match \
    --default-index https://mirrors.aliyun.com/pypi/simple/ \
    --index https://download.pytorch.org/whl/cpu \
    --from "mflowy[modeling]" \
    mcpSrv

# 轻量分析（仅 [stats]，无 torch，无需 CPU 索引参数）
uvx --from "mflowy[stats]" cmd data_profile '{"file_path": "..."}'

# 或常规安装（pip / uv pip）
pip install "mflowy[modeling]"
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
        "--index", "https://download.pytorch.org/whl/cpu",  // [modeling] 需要
        "--from", "mflowy[modeling]",
        "mcpSrv"
      ]
    }
  }
}
```

### 方式二：源码（开发 / 贡献）

```bash
git clone https://github.com/ifoodsci-ai/mflowy.git && cd mflowy
uv sync --all-extras --all-groups

uv run cmd list_modules                                       # 查看支持的步骤及模块列表（base，无数据栈）
uv run --extra stats cmd data_profile '{"file_path": "..."}'  # 数据分析工具
uv run mcpSrv                                                  # MCP server（stdio）
```

更多开发命令（测试 / lint / 构建）见 [AGENTS.md](AGENTS.md) 或 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 环境变量

| 变量                    | 用途                                                                                           | 示例                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------ |
| `MLFLOW_TRACKING_URI` | Tracking server URI（未设置时 workflow 与查询工具同落固定库 `~/.mflowy/mlflow.db`） | `postgresql://user:pwd@host:5432/mlflow` |
| `MFLOWY_JOB_PROVIDER` | JobProvider 解析：`local`（默认）或 `module:Class`（自定义实现）                           | `my_pkg.job_provider:MyRemoteProvider`   |
| `PYTHONPATH`          | 自定义 JobProvider 模块的包根                                                                  | `/srv/my-provider`                       |
| `MFLOWY_TELEMETRY`   | 遥测模式：`ask`（默认，首次工具调用时询问）/ `on` / `off`（显式设置覆盖 settings.json，见下方「遥测」） | `on`                                  |

### MCP 客户端配置示例

见上方方式一内的 JSON 示例，或直接参考 [`.mcp.json.example`](.mcp.json.example)（PyPI 形态）。

### 启动说明

- **入口名 `mcpSrv`**：刻意避开 mcp SDK 自带的同名 `mcp` CLI（`mcp.cli:app`）——uvx 解析 `mcp` 命令时可能命中 SDK 侧导致启动失败
- **extras 内联在 `--from` spec**：uvx 的 `--extra` 需新版 uv，内联写法兼容性最好
- **离线分发用 wheel**：`make build-whl` 产出 `dist/mflowy-<version>-py3-none-any.whl`，`--from "mflowy[modeling] @ file://<whl绝对路径>"` 替代包名（K8s 镜像构建/内网场景）
- **torch CPU 索引**（`--index` pytorch-cpu + `--index-strategy unsafe-best-match`）为 [modeling] 必需：uvx 不读 pyproject 的 `[tool.uv.sources]`，缺省时 torch 解析为 CUDA 全家桶（2–3GB）；`unsafe-best-match` 须与 pytorch 索引同用，否则 first-index 策略会因该索引上的旧版 requests 解析失败

## 遥测（Telemetry）

MCP 工具调用诊断采集，同意制、默认 `ask`，端点不可达时完全透明不影响工具调用，仅覆盖 MCP 入口。隐私契约见 [PRIVACY.md](PRIVACY.md)，接入与配置详情见 [packages/mcp/README.md](packages/mcp/README.md)。

## 贡献

欢迎任何形式的贡献（功能、修复、文档、案例）。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)（开发流程与约定）、[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)、[PRIVACY.md](PRIVACY.md)（遥测隐私契约）与 [SECURITY.md](SECURITY.md)（漏洞披露）。

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 文档

- [架构与贡献指南](AGENTS.md)
- [研究流方法论](docs/research-flow.md)（agent 可直接采用的工作规约）
- [路线图](docs/roadmap.md)
- [DAG 内核：设计哲学与架构](packages/driver/README.md)
- [内置能力目录（含第三方插件指南）](packages/builtin_plugins/README.md)
- [MCP 层：工具 / 远程执行 / 遥测](packages/mcp/README.md)
- [共享工具层](packages/utils/README.md)
