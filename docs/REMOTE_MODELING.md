# Remote Modeling（远程执行）

MFlowy 不内置绑定任何远程执行平台。compute 工具的执行策略由
**JobProvider 契约**抽象（`packages/mcp/src/mflowy/mcp/job_provider/protocol.py`）。

## 工具分工

| 工具类别 | 执行位置 | 说明 |
| -------- | -------- | ---- |
| 分析类（`data_profile` / `eda` / `infer_task_type_by_statistic`） | **始终本地** | 轻量、无跨环境状态，不经 JobProvider |
| 建模类（`modeling` / `explanation` / `predict` / `inverse_optimization`） | **JobProvider 委派** | 内置 `local` 实现（LocalJobProvider，[modeling] extra 直调编排） |

## 接入自定义执行环境

实现 `JobProvider` 协议的 **4 个类型化方法**（每个方法带 `headers: Mapping[str, str] | None`
透传请求元数据，MCP 客户端经 `call_tool(meta={...})` 携带），即可把建模工具委派到
任意执行环境——K8s Job、远程集群、Serverless 均可。方法签名以
`packages/mcp/src/mflowy/mcp/job_provider/protocol.py` 为准（本文不复述，防漂移）；契约行为由
`tests/mcp/test_job_provider.py` / `test_tools_headers.py` 锁定。

启用方式：

```bash
PYTHONPATH=/srv/my-provider \
MFLOWY_JOB_PROVIDER=my_pkg.job_provider:MyRemoteProvider \
uvx --from "mflowy[modeling]" mcpSrv   # 或 wheel 形态：--from "mflowy[modeling] @ file://<whl>"
```

- `MFLOWY_JOB_PROVIDER`：`local`（默认）或 `module:Class`
- 模块经宿主 `sys.path` 解析，provider 以标准 Python 包交付、依赖由其自身
  pyproject 声明

## 规划预留

底层资源装载的 `JobProvisioner` 契约（mount / create / unmount / get_output）
为规划预留，尚未实现——见 [roadmap.md](roadmap.md)。
