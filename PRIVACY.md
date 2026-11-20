# 隐私契约（Telemetry Privacy）

MFlowy 的 CLI 遥测遵循「**明确同意、数据只发往配置端点（默认项目官方遥测后端）**」原则。本文是可执行的隐私契约；任何削弱它的改动都视为隐私回归。

## 采集什么

仅 CLI 入口（`mflowy`）的工具调用诊断信息，由内置 OTLP JSON 发送器采集（agentcat 与 MCP SDK 已随 #243 退役——无第三方遥测库、无外部注入参数）：

- **工具名** 与每次调用的**完整输入参数、输出结果**（OTLP span 属性 `mflowy.parameters`（工具入参）/ `mflowy.response`（结构化结果优先），全量 JSON 不截断；脱敏例外见下——其余风险见下方「风险边界」）
- **调用意图与任务关联**：`--telemetry-intent`（调用方自述「为什么调这个工具」）与 `--telemetry-session-id`（跨调用任务关联）——均为 CLI 元旗标、调用方主动注入，不进工具签名
- 错误类型与消息

遥测副本的四条脱敏例外（只作用于上报副本，工具真实返回与入参零影响；实现见
`packages/app/mflowy/app/telemetry.py`）：

1. **UUID 全局掩码**：事件内全部字符串（参数 / 响应 / 意图 / 错误串）中的 canonical UUID（带连字符）替换为 `<uuid>`——平台 experiment / job 等标识符不可关联到平台账号；mlflow run_id（32 位无连字符 hex）不受影响，本地关联保持。
2. **login 标识字段剔除**：`login` 工具遥测副本的入参与返回中剔除 `user_id` / `device_session_id` / `device_name`（缺省值为 hostname，属机器标识）。
3. **logs 文本剔除**：`get_platform_job_logs` 遥测副本中整字段剔除 `logs`（日志自由文本可能含路径/数据样本片段/任意标识）——`{job_id, status, terminal, tail_lines}` 照常上报。
4. **会话锚不采集**：工具入参中的 `observation_json`（状态文件本机绝对路径）与旧键 `session_id` 不进遥测副本——遥测 session_id 仅由调用方经元旗标注入，不做锚自动提取。

不采集：`mflowy-job` 容器入口与直接 import 入口；文件内容读取；身份标识符——机器标识、用户标识按构造不进遥测（例外 1 的 UUID 掩码 + 例外 2/3 的字段剔除兜住已知标识面，自由文本措辞风险见「风险边界」）。设备身份例外披露：上报密钥绑定的 `actor_id` 由 machine-id 派生（见「设备密钥」节）——密钥绑定身份，非工具调用内容。

## 数据去哪里

**默认**发往项目官方遥测后端 `https://telemetry.ifoodsci.com`（`DEFAULT_ENDPOINT` 常量，见
`packages/app/mflowy/app/telemetry.py`，TLS 加密传输）；环境变量
`MFLOWY_TELEMETRY_ENDPOINT` 可将端点指向自建接收网关（自包含部署见 `docker/telemetry/`）——
设备注册与 OTLP 导出共用同一端点解析。同意遥测即表示你接受上述数据进入生效端点；
不信任时 `MFLOWY_TELEMETRY=off`。

发送为 fire-and-forget（后台线程、短超时、失败静默）——端点不可达完全透明，不影响工具调用与退出码。

## 同意模型（合规）

`~/.mflowy/settings.json` 的 `telemetry` 键（bool）+ env `MFLOWY_TELEMETRY=on|off`：

| 状态 | 行为 |
| --- | --- |
| `telemetry` 键缺省 + TTY | 首次调用时终端 y/n 询问，**答案持久化**到该键（此后不再问） |
| `telemetry` 键缺省 + 非 TTY（agent 场景） | 默认**开**，stderr 打一行关闭指引（不落盘） |
| `true` / `false` | 按键值生效（`false` = 关） |
| env `on` / `off` | 显式设置即视为操作员授权/拒绝，**无条件优先于** settings.json（只影响本次及后续启动的生效值，不回写文件；K8s/CI 等无交互场景的唯一入口） |
| `mflowy login` 成功 | **无条件写 `telemetry=true`**（登录即同意） |

## 设备密钥

开启遥测后，`~/.mflowy/api.key`（0600）保存本设备的上报秘密——**由 mflowy 在本机生成**（服务端只存 hash，注册时经 TLS 上传一次；401/403 自动重签重绑）。设备身份取 machine-id（跨平台硬件级标识，进入 `actor_id`）。删除该文件即可在下次上报时以新秘密重新注册绑定。

## 关闭 / 撤回同意

- 环境变量 `MFLOWY_TELEMETRY=off`（最高优先级，立即生效于下次启动）
- 或编辑 `~/.mflowy/settings.json`：`{"telemetry": false}` 后重启

## 风险边界（务必阅读）

输入输出**全量采集**（仅上列四条脱敏例外）意味着：工具调用输入与输出——比如数据文件路径、列名、预测结果摘要等——会随 span
进入生效遥测后端（默认 telemetry.ifoodsci.com，TLS 传输）；意图文本由调用方 LLM 生成，措辞不可控，
自由文本若自述身份信息不在掩码保证面内。**同意前请确认你接受上述内容进入该后端**；
不接受则拒绝首问（TTY 答n）或 `MFLOWY_TELEMETRY=off`。

实现见 `packages/app/mflowy/app/telemetry.py`。
