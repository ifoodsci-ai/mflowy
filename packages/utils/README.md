# mflowy-utils — 共享工具层

MFlowy 全家共享的底层工具，无业务语义：

| 职责 | 模块 | 说明 |
|------|------|------|
| 实验追踪封装 | `mlflow.py` | tracking URI / experiment 显式传递（隔离 mlflow 进程级全局）、tag 与 figure/table 落盘 helper |
| 模板 | `jinja.py` | 沙箱化 Jinja2 环境（SandboxedEnvironment + StrictUndefined），YAML 模板渲染 |
| 观测基础设施 | `logging.py` `capture.py` | stderr 日志分级、stdout 按任务捕获（通道边界的实现侧） |
| 脚本安全 | `python_script_security_scan.py` | `load.python` / `clean.python` 的 AST 安全门（禁 os/sys/subprocess 等危险调用） |
| 内容指纹 | `file.py` | 文件读写 + `sha256_of` / `fingerprint_tags`（run 级数据指纹 tag 的实现侧） |
| 任务目录锚定 | `path.py` | ContextVar 每请求/每 asyncio task 隔离——MCP 并发分发下进行中 DAG 的产物解析不串目录 |
| 通用 | `wraps.py` `utils.py` | 同步/静默装饰器、时间戳 |

主文档与导航见[仓库根 README](../../README.md)。
