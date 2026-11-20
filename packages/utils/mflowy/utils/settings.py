"""全局配置根 ``~/.mflowy/settings.json`` —— 键值面的单源读写件。

消费方（#243）：遥测同意键 ``telemetry``（app 层）、执行模式键 ``job_provider``
（model 域解析序 env > settings > local）。写操作保留无关键、tmp+rename 原子落盘
（红线 3）；读操作对缺文件按未配置处理（返回 None）、对**坏文件**（在盘但不可读/
坏 JSON/非对象）记 stderr warning 后同样按未配置——静默丢弃显式化（红线 1 折衷：
读失败不炸调用方，但绝不无声）。
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from .wraps import synchronized

logger = logging.getLogger(__name__)


def settings_path() -> Path:
    """配置根路径（测试经 HOME 隔离；不做 cache——path 求值廉价且免去缓存清理面）。"""
    return Path.home() / ".mflowy" / "settings.json"


def read_settings() -> dict:
    """读整个配置对象；缺文件/坏 JSON/非对象 → 空 dict（坏文件记 warning，调用方按键取值）。"""
    path = settings_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("settings.json 在盘但不可读（%s）——按未配置处理：%s", exc, path)
        return {}
    if not isinstance(data, dict):
        logger.warning("settings.json 顶层不是对象——按未配置处理：%s", path)
        return {}
    return data


def read_key(key: str):
    """取单键；缺省 None（三态语义由调用方解释：None=未配置）。"""
    return read_settings().get(key, None)


@synchronized()
def write_key(key: str, value) -> None:
    """写单键：保留无关配置键，tmp+rename 原子落盘。"""
    path = settings_path()
    data = read_settings()
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".settings.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp, path)  # 同目录 rename：原子（读者只见旧或新，无半态）
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
