# mflowy-utils — 共享工具层

MFlowy 全家共享的底层工具，无业务语义：

| 职责 | 模块 | 说明 |
|------|------|------|
| 实验追踪封装 | `mlflow.py` | tracking URI / experiment 显式传递（隔离 mlflow 进程级全局）、tag 与 figure/table 落盘 helper |
| 模板 | `jinja.py` | 沙箱化 Jinja2 环境（SandboxedEnvironment + StrictUndefined），YAML 模板渲染 |
| 观测基础设施 | `logging.py` `capture.py` | stderr 日志分级、stdout 按任务捕获（通道边界的实现侧） |
| 脚本安全 | `python_script_security_scan.py` | `load.python` / `clean.python` 的 AST 安全门（禁 os/sys/subprocess 等危险调用） |
| 通用 | `file.py` `path.py` `wraps.py` `constants.py` | 文件读写、路径上下文、同步/静默装饰器 |

主文档与导航见[仓库根 README](../../README.md)。
