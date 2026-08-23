# 隐私契约（Telemetry Privacy）

MFlowy 的 MCP 遥测遵循「**默认询问、明确同意、数据只发往固定端点（项目官方遥测后端）**」原则。本文是可执行的隐私契约；任何削弱它的改动都视为隐私回归。

## 采集什么

仅 MCP 入口（`mcpSrv`）的工具调用诊断信息，由 [agentcat](https://pypi.org/project/agentcat/) 以纯 OTLP 模式采集：

- **工具名** 与每次调用的**完整输入参数、输出结果**（OTLP span 属性 `mcp.parameters`（工具入参）/ `mcp.response`（结构化结果优先），全量 JSON 不截断不脱敏——因此见下方「风险边界」）
- **调用意图**：agentcat 向工具 schema 注入 `context` 参数，由调用方 LLM 自述「为什么调这个工具」（handler 执行前即剥离，业务代码无感知）
- 错误与异常栈

不采集：`cmd` JSON runner 与直接 import 入口（不经 MCP 协议）；机器标识、用户标识、文件内容读取。

## 数据去哪里

**只**发往项目官方遥测后端 `https://telemetry.ifoodsci.com`（`DEFAULT_ENDPOINT` 常量，见
`src/mflowy/mcp/telemetry.py`，TLS 加密传输）。
同意遥测即表示你接受上述数据进入该后端；不信任时 decline / `MFLOWY_TELEMETRY=off`，
自建接收器需改源码常量（docker/telemetry 自包含部署）。

不经 AgentCat 云（无项目 ID、无账号）；agentcat 自身的匿名诊断上报已通过 `disable_diagnostics=True` 关闭。

## 同意模型（合规）

`MFLOWY_TELEMETRY=ask|on|off`，**默认 `ask`**：

| 模式 | 行为 |
| --- | --- |
| `ask`（默认） | 首次工具调用时经 MCP elicitation 询问（无表单字段，仅 accept/decline/cancel）；accept 与 cancel 均视为同意，decline 视为拒绝——**三者都持久化**到 `~/.mflowy/settings.json` 的 `telemetry` 键（bool），此后不再询问 |
| `on` / `off` | 显式设置即视为操作员授权/拒绝，**无条件优先于** settings.json（只影响本次及后续启动的生效值，不回写文件；K8s/CI 等无交互场景的唯一入口） |

客户端不支持 elicitation（提问失败）时：本次调用不采集、不落盘（下次调用会再问），工具调用不受任何影响。

## 设备密钥

开启遥测后，`~/.mflowy/api.key`（0600）保存本设备的上报秘密——**由 mflowy 在本机生成**（服务端只存 hash，注册时经 TLS 上传一次）。设备身份取 machine-id（跨平台硬件级标识，进入 `actor_id`）。删除该文件即可在下次上报时以新秘密重新注册绑定。

## 关闭 / 撤回同意

- 环境变量 `MFLOWY_TELEMETRY=off`（最高优先级，立即生效于下次启动）
- 或编辑 `~/.mflowy/settings.json`：`{"telemetry": false}` 后重启

## 风险边界（务必阅读）

输入输出**全量采集**意味着：数据文件路径、列名、预测结果摘要等会随 span 进入官方遥测后端
（telemetry.ifoodsci.com，TLS 传输）；意图文本由调用方 LLM 生成，措辞不可控。
**同意前请确认你接受上述内容进入官方后端**；不接受则 decline 或 `MFLOWY_TELEMETRY=off`。

实现见 `src/mflowy/mcp/telemetry.py`。
