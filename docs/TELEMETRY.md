# Telemetry（遥测）

MCP 工具调用诊断采集：工具名、完整输入输出、调用意图。由 [agentcat](https://pypi.org/project/agentcat/)
以**纯 OTLP 模式**直出固定端点，不经 AgentCat 云（无项目 ID、无账号，
agentcat 自身匿名诊断已通过 `disable_diagnostics=True` 关闭）。

## 同意模型

- **默认 `ask`**：首次工具调用时经 MCP elicitation 询问；accept/cancel 视为同意、
  decline 视为拒绝，选择持久化到 `~/.mflowy/settings.json`，永不再问
- `MFLOWY_TELEMETRY=on|off` 显式覆盖（容器 / CI 等无交互场景的唯一入口），
  无条件优先于 settings.json
- 客户端不支持 elicitation 时：本次不采集、不落盘，下次再问，工具调用不受影响

## 数据去向

- 遥测后端：`https://telemetry.ifoodsci.com`（项目官方遥测网关，TLS；
  自建部署见 [docker/telemetry/](../docker/telemetry/)）
- 设备密钥：`~/.mflowy/api.key`（0600）本机生成，服务端只存 hash

## 边界

- 端点不可达完全透明：不阻塞、不影响工具调用，静默重试 / 丢弃
- 仅覆盖 MCP 入口（`mcpSrv`）；`cmd` runner 与直接 import 不采集
- 采集范围、撤回方式、风险边界等隐私契约见 [PRIVACY.md](../PRIVACY.md)

## 参考

- 实现：`src/mflowy/mcp/telemetry.py`（唯一接线点 `wire_agentcat`，禁止 per-tool 打点）
