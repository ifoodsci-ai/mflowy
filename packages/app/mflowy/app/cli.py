"""mflowy CLI — 旗标从 pyfunc pydantic 签名编译（#243：SKILLS+CLI 替代 SKILLS+MCP）。

单源纪律：签名=约束=帮助=校验（同一台 pydantic 机器）——

- **旗标编译**：每个参数 → ``--kebab-case`` 旗标；``Field description`` → help；
  签名默认值 → help 缺省标注。标量 → 类型化旗标（int/float/str）；bool →
  ``--flag/--no-flag``；``list``/``dict``/嵌套 pydantic 模型 → JSON 字符串旗标
  （CLI 层仅 ``json.loads``，零语义校验）。
- **运行时校验**：动态 ``create_model``（``extra="forbid"``）对合并后 kwargs 做
  pydantic 校验 + coerce——旗标层零校验，非法值在此 fail-loud（exit 2）。
- **逃生舱**：``--kwargs-json '<JSON 对象>'`` 全量传参（jobwrap 先例）；与具名
  旗标同键 = 参数来源冲突拒绝（红线 1，绝不静默覆写）。
- **遥测元旗标**：``--telemetry-session-id`` / ``--telemetry-intent``——CLI 层剥离、
  不进 pyfunc 签名，调用方主动注入。

帮助面：``mflowy -h`` = 按所属域分组的工具清单 + 一行「何时使用」介；
``mflowy <tool> -h`` = usage / 何时（docstring 全文）/ options 三段。
输出与退出码继承 runner 既有契约（查询类→stdout JSON、DAG 类→流式上屏、
JobOutcome/DAGResult→退出码映射）；分发与退出码单源 = ``runner.run_tool``。
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import types
import typing
from typing import Any, get_args, get_origin, get_type_hints

# 旗标缺省统一 SUPPRESS：namespace 只含显式传入的旗标（缺省与必填校验都交给
# pydantic 运行时层），显式面 = --kwargs-json 同键冲突检测的事实源。
_SUPPRESS = argparse.SUPPRESS


def _kebab(name: str) -> str:
    return "--" + name.replace("_", "-")


def _unwrap_annotated(annotation: Any) -> tuple[Any, list[Any]]:
    """剥 Annotated 层：返回 (底层类型, 元数据列表)——Field 描述即旗标 help 单源。"""
    metadata: list[Any] = []
    while hasattr(annotation, "__metadata__"):
        metadata.extend(annotation.__metadata__)
        annotation = annotation.__origin__
    return annotation, metadata


def _optional_base(tp: Any) -> tuple[Any, bool]:
    """剥 Optional（``X | None`` / ``Optional[X]``）：返回 (底层类型, 是否可选)。"""
    origin = get_origin(tp)
    if origin is types.UnionType or origin is typing.Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return tp, False


def _classify(tp: Any) -> str:
    """底层类型 → 旗标形态：bool / int / float / str / json（复杂类型走 JSON 串）。"""
    from pydantic import BaseModel

    base, _ = _optional_base(tp)
    if base is bool:
        return "bool"
    if base in (int, float, str):
        return base.__name__
    if inspect.isclass(base) and issubclass(base, BaseModel):
        return "json"
    return "json"  # list/dict/联合/透传第三方类型：JSON 字符串旗标


def _expand_model(tp: Any) -> str:
    """嵌套 pydantic 模型展开（#233 验收线：``-h`` 须揭示 --prior 类嵌套 schema）。

    模型注解的参数在 help 里逐字段展开（字段名/类型/必填标记/描述首句），
    ``$defs`` 子模型引用递归展开（递归引用展开一次即止）；``dict`` 注解的嵌套
    面由 Field description 自足承载（当前工具面形态）。
    """
    from pydantic import BaseModel

    lines: list[str] = []
    seen: set[type] = set()

    def _fmt(name: str, spec: dict, required: set, pad: str) -> str | None:
        mark = "" if name in required else "?"
        typ = spec.get("type") or spec.get("$ref", "").rsplit("/", 1)[-1] or "any"
        desc = (spec.get("description") or "").split("。")[0]
        return f"{pad}{name}{mark}: {typ}" + (f" — {desc}" if desc else "")

    def _walk(cls: type[BaseModel], depth: int) -> None:
        if cls in seen:  # 递归引用：展开一次即止
            return
        seen.add(cls)
        schema = cls.model_json_schema()
        required = set(schema.get("required", []))
        pad = "  " * depth
        for name, spec in schema.get("properties", {}).items():
            line = _fmt(name, spec, required, pad)
            if line:
                lines.append(line)
            if not spec.get("$ref"):
                continue
            # 子模型引用：优先经字段注解直取模型类递归展开（任意深度）；注解不可
            # 直取时退回 $defs 定义展开属性面（一层）
            ann = getattr(cls.model_fields.get(name, None), "annotation", None)
            base_ann, _ = _optional_base(ann) if ann is not None else (ann, False)
            if inspect.isclass(base_ann) and issubclass(base_ann, BaseModel):
                _walk(base_ann, depth + 1)
                continue
            sub = schema.get("$defs", {}).get(spec["$ref"].rsplit("/", 1)[-1], {})
            for sub_field, sub_spec in sub.get("properties", {}).items():
                sub_line = _fmt(sub_field, sub_spec, set(sub.get("required", [])), "  " * (depth + 1))
                if sub_line:
                    lines.append(sub_line)

    base, _ = _optional_base(tp)
    if inspect.isclass(base) and issubclass(base, BaseModel):
        _walk(base, 1)
    return "\n".join(lines)


def _field_help(param: inspect.Parameter, annotation: Any) -> tuple[str, str | None]:
    """参数 → (help 文案, argparse type)。help = Field description + 类型/缺省标注。"""
    from pydantic.fields import FieldInfo

    tp, metadata = _unwrap_annotated(annotation)
    description = ""
    for meta in metadata:
        if isinstance(meta, FieldInfo):
            description = meta.description or ""
            break
    kind = _classify(tp)
    optional = param.default is not inspect.Parameter.empty
    default_repr = "" if param.default is inspect.Parameter.empty else repr(param.default)
    tag = (
        f"必填 {kind.upper() if kind == 'json' else kind}"
        if not optional
        else (f"{kind.upper() if kind == 'json' else kind}，缺省 {default_repr}")
    )
    help_text = f"{description}\n[{tag}]" if description else f"[{tag}]"
    if kind == "json":
        expanded = _expand_model(tp)
        if expanded:
            help_text += f"\n{expanded}"
    arg_type: str | None = {"int": int, "float": float, "str": str}.get(kind)
    return help_text, arg_type


def build_tool_parser(fn, prog: str) -> argparse.ArgumentParser:
    """pyfunc → 单工具 ArgumentParser（旗标面全量编译 + 逃生舱 + 遥测元旗标）。"""
    parser = argparse.ArgumentParser(
        prog=prog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(inspect.getdoc(fn) or fn.__name__),
        epilog="复杂类型旗标收 JSON 字符串（--prior '{...}'）；--kwargs-json 可全量传参。",
    )
    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        help_text, arg_type = _field_help(param, annotation)
        kind = _classify(_unwrap_annotated(annotation)[0])
        # required 不在 argparse 层强制（--kwargs-json 逃生舱须能独立满足必填）——
        # 缺参由 pydantic 运行时校验 fail-loud（exit 2，help 里【必填】标注常在）
        if kind == "bool":
            parser.add_argument(
                _kebab(name),
                help=help_text,
                default=_SUPPRESS,
                action=argparse.BooleanOptionalAction,
            )
        else:
            # 标量 argparse 类型转换；json 形态收原始串（metavar 提示）——type=None
            # 是「不转换」的合法形态（argparse 类型重载在 pyright 面按 kwargs 显式分支）
            if arg_type is not None:
                parser.add_argument(_kebab(name), help=help_text, default=_SUPPRESS, type=arg_type)
            else:
                parser.add_argument(_kebab(name), help=help_text, default=_SUPPRESS, metavar="JSON")
    parser.add_argument(
        "--kwargs-json",
        metavar="JSON",
        help="全量参数 JSON 对象（逃生舱）——与具名旗标同键 = 参数来源冲突拒绝",
        default=_SUPPRESS,
    )
    parser.add_argument(
        _kebab("telemetry_session_id"),
        dest="telemetry_session_id",
        help="遥测会话标识（调用方注入；CLI 层剥离不进工具签名）",
        default=_SUPPRESS,
    )
    parser.add_argument(
        _kebab("telemetry_intent"),
        dest="telemetry_intent",
        help="遥测调用意图（调用方注入；CLI 层剥离不进工具签名）",
        default=_SUPPRESS,
    )
    return parser


# ── 校验（单源 = pydantic；旗标层零校验）


def validated_kwargs(fn, kwargs: dict) -> dict:
    """签名 → 动态模型 → pydantic 校验 + coerce（未知键 fail-loud）。"""
    from pydantic import ConfigDict, ValidationError, create_model

    hints = get_type_hints(fn, include_extras=True)
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        ann = hints.get(name, param.annotation)
        fields[name] = (ann, ... if param.default is inspect.Parameter.empty else param.default)
    model = create_model("ToolArgs", __config__=ConfigDict(extra="forbid", protected_namespaces=()), **fields)
    try:
        return model(**kwargs).model_dump()
    except ValidationError as e:
        print(f"参数校验失败（{fn.__name__}）:\n{e}", file=sys.stderr)
        raise SystemExit(2) from None


def collect_args(fn, ns: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[dict, str | None, str | None]:
    """namespace → (kwargs, session_id, intent)：JSON 串解码、元旗标剥离、来源冲突拒绝。"""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    explicit: dict[str, Any] = {}
    for name in sig.parameters:
        if hasattr(ns, name):
            value = getattr(ns, name)
            kind = _classify(_unwrap_annotated(hints.get(name, sig.parameters[name].annotation))[0])
            if kind == "json" and isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as e:
                    parser.error(f"{_kebab(name)} 值非合法 JSON：{e}")
            explicit[name] = value
    if hasattr(ns, "kwargs_json"):
        try:
            extra = json.loads(ns.kwargs_json)
        except json.JSONDecodeError as e:
            parser.error(f"--kwargs-json 非 JSON：{e}")
        if not isinstance(extra, dict):
            parser.error(f"--kwargs-json 须为 JSON 对象（got {type(extra).__name__}）")
        clash = sorted(set(explicit) & set(extra))
        if clash:
            parser.error(f"参数来源冲突：具名旗标与 --kwargs-json 同键 {clash}——同参只允许一个来源")
        explicit.update(extra)
    session_id = getattr(ns, "telemetry_session_id", None)
    intent = getattr(ns, "telemetry_intent", None)
    return explicit, session_id, intent


# ── 工具面（分组清单 + 一行介）


_GROUPS: list[tuple[str, str]] = [
    ("mflowy.app", "平台（词表/观测/登录/平台 API）"),
    ("mflowy.data", "数据清洗域（本地分析 + 清洗会话）"),
    ("mflowy.model", "建模域（compute + 建模会话 + 本地分析 + YAML 预检）"),
    ("mflowy.bo", "BO 推荐域"),
]


def _group_label(ep) -> str:
    module = ep.value.split(":")[0]
    for prefix, label in _GROUPS:
        if module.startswith(prefix + ".") or module == prefix:
            return label
    return "其他"


def _one_liner(fn) -> str:
    """「何时使用」首条（含缩进折行折叠；无则 docstring 首段）——mflowy -h 清单的一行介。"""
    doc = inspect.getdoc(fn) or ""
    lines = doc.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"何时使用 \w+：\s*$", line):
            joined: list[str] = []
            seen_bullet = False
            for cont in lines[i + 1 :]:
                if cont.lstrip().startswith("- "):
                    if seen_bullet:
                        break  # 首条 bullet 结束（下一条开始）
                    seen_bullet = True
                    joined.append(cont.strip()[2:])
                elif not cont.strip():
                    break  # 空行 = 段落结束
                elif cont[:1] in (" ", "\t"):
                    joined.append(cont.strip())  # 相对缩进折行折叠
                else:
                    break  # 顶格非 bullet = 新段落
            if joined:
                return " ".join(joined)
            break
    return doc.split("\n\n")[0].replace("\n", " ")


def _print_grouped_help() -> None:
    from .tools import _tool_entry_points

    eps = sorted(_tool_entry_points(), key=lambda e: e.name)
    grouped: dict[str, list[tuple[str, Any]]] = {}
    for ep in eps:
        grouped.setdefault(_group_label(ep), []).append((ep.name, ep.load()))
    print("mflowy — MFlowy 能力 CLI（工具扁平命名空间）")
    print()
    print("用法: mflowy <tool> [旗标]        # 旗标从工具签名编译；mflowy <tool> -h 看全参与用法")
    print("      mflowy <tool> --kwargs-json '<JSON 对象>'   # 全量传参逃生舱")
    print()
    for label in [lbl for _, lbl in _GROUPS] + ["其他"]:
        entries = grouped.get(label)
        if not entries:
            continue
        print(f"{label}（{len(entries)}）:")
        width = max(len(n) for n, _ in entries)
        for name, fn in entries:
            print(f"  {name.ljust(width)}  {_one_liner(fn)}")
        print()


def _resolve_tool(tool_name: str):
    from .tools import _tool_entry_points

    try:
        eps = _tool_entry_points()
    except RuntimeError as e:  # 撞名 fail-loud（守卫单源住 tools.py）
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    table = {ep.name: ep.load() for ep in eps}
    fn = table.get(tool_name)
    if fn is None:
        print(f"Unknown tool: {tool_name}\n可选工具（mflowy --help 看分组全表）: {sorted(table)}", file=sys.stderr)
        raise SystemExit(1)
    return fn


def main(argv: list[str] | None = None) -> None:
    from mflowy.utils.logging import setup

    setup()  # 日志绑定 stderr（stdout 只走业务数据）

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _print_grouped_help()
        return

    tool_name = argv[0].replace("-", "_")
    fn = _resolve_tool(tool_name)
    parser = build_tool_parser(fn, prog=f"mflowy {argv[0]}")
    ns = parser.parse_args(argv[1:])
    kwargs, session_id, intent = collect_args(fn, ns, parser)
    validated = validated_kwargs(fn, kwargs)

    from . import telemetry
    from .runner import run_tool

    def _observe(name: str, kw: dict, result, error: BaseException | None) -> None:
        telemetry.report_call(name, kw, result, error, session_id=session_id, intent=intent)

    sys.exit(run_tool(tool_name, validated, observe=_observe))


if __name__ == "__main__":
    main()
