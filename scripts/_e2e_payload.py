"""e2e 工具调用 JSON 载荷构造（#66 收口后 Makefile 不再手拼 JSON）。

用法：``_e2e_payload.py <steps.yaml> [--stage <dir>] [key=value ...]`` → stdout 单行
JSON（``modeling_steps`` = YAML 明文 + 附加字段直拼；json.dumps 恒单行）。

- 明文边界注意：``modeling_steps`` 无自身文件路径，load 步相对 ``source``（如
  ``diabetes.py:load``）会锚到进程 cwd——本脚本持有 YAML 路径，负责把相对 source
  基于 YAML 所在目录绝对化（``absolutize_load_sources``，_e2e_bench 复用）。
- ``--stage <dir>``：生成器复制出仓再改写 source 指向副本——工件锚 stage 目录，
  零仓库产物（M10）。
"""

import json
import pathlib
import re
import shutil
import sys


def absolutize_load_sources(text: str, base_dir) -> str:
    """相对 ``source: xx.py[:func]`` → 基于 YAML 所在目录的绝对路径。"""

    def _sub(m: re.Match) -> str:
        return f"source: {(pathlib.Path(base_dir) / m.group(1)).resolve()}"

    return re.sub(r"source: (?!/)([\w./-]+\.py(?::\w+)?)", _sub, text)


def stage_load_sources(text: str, stage_dir) -> str:
    """绝对 ``source: /abs/xx.py[:func]`` 的生成器复制到 stage 目录并改写引用。"""

    def _sub(m: re.Match) -> str:
        src = pathlib.Path(m.group(1))
        staged = pathlib.Path(stage_dir) / src.name
        if not staged.exists():
            shutil.copy2(src, staged)
        return f"source: {staged}{m.group(2) or ''}"

    return re.sub(r"source: (/[\w./-]+\.py)(:\w+)?", _sub, text)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法: _e2e_payload.py <steps.yaml> [--stage <dir>] [key=value ...]")

    yaml_path = pathlib.Path(sys.argv[1])
    stage = None
    extras: list[str] = []
    args = iter(sys.argv[2:])
    for arg in args:
        if arg == "--stage":
            stage = next(args, None) or ""
        else:
            extras.append(arg)

    steps_text = absolutize_load_sources(yaml_path.read_text(encoding="utf-8"), yaml_path.parent)
    if stage:
        steps_text = stage_load_sources(steps_text, stage)

    payload = {"modeling_steps": steps_text}
    for kv in extras:
        key, _, value = kv.partition("=")
        if not key:
            raise SystemExit(f"附加字段须为 key=value 形式，got {kv!r}")
        payload[key] = value

    print(json.dumps(payload))


if __name__ == "__main__":
    main()
