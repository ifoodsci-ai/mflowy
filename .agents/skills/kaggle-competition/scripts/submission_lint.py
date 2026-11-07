"""提交包 preflight 校验（竞赛外环回路的 lint 动作——活脚本，飞轮沉积处）。

设计依据：本 skill MDP.md——平台校验器是黑箱，本脚本把已实证的约束本地化，
不把提交预算（每日 5 次）花在可本地判定的失败上。规则表与实证出处见
references/package.md；新发现的平台 error_description 回流两步：
① package.md 规则表补一行 ② 本脚本补实现（约束发现 → 规则沉淀的飞轮）。

跨赛通用硬规则（内置）：
  ERROR  zip 内 .csv/.parquet 恰好一个（实证：多文件/零文件两次 ERROR 夹逼收敛）
  ERROR  唯一 csv 含字面 id 列（--id-column 可改）；给了 --expect-rows 时行数须相等
  ERROR  --require 传入的必备交付物缺件（SPEC 阶段按赛题 deliverables 清单给定）
  ERROR  notebook 非法 JSON 或无 cells
  ERROR  包指纹与 submissions.json 中曾 ERROR 的提交相同——不重试同包（守卫沉在代码）
  WARN   包体 > 100MB / 构建残渣（__pycache__ .mlruns .DS_Store）
  WARN   包指纹与曾成功（complete）的提交相同——无增量
  WARN   未给 --expect-rows：行数规则跳过（host solution 行数须从赛题标定）

用法：
  python submission_lint.py <submission_package.zip> [--submissions submissions.json]
      [--id-column id] [--expect-rows N] [--require f1,f2,...] [--print-fingerprint]
退出码：0 = 可提交；1 = 存在 ERROR（build 收尾可直接消费）。
submissions.json = 回路账本（工作目录维护，非本脚本状态）：
  [{"fingerprint": "...", "ref": 57066480, "status": "error|complete", "ts": "..."}, ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

JUNK_MARKERS = ("__pycache__", ".mlruns", ".DS_Store")
SIZE_WARN = 100 * 1024 * 1024


def fingerprint(zpath: Path) -> str:
    """内容指纹：对 zip 内 (arcname, 文件 sha256) 排序级联再哈希。

    与 zip 时间戳/压缩参数无关——重打包同内容同指纹，改内容即换指纹，
    这是重试守卫能判「同一个包」的前提。
    """
    h = hashlib.sha256()
    with zipfile.ZipFile(zpath) as z:
        for name in sorted(z.namelist()):
            fh = hashlib.sha256(z.read(name)).hexdigest()
            h.update(f"{name}:{fh}\n".encode())
    return h.hexdigest()[:16]


def lint(zpath: Path, subs: Path | None, id_column: str, expect_rows: int | None, require: list[str]) -> int:
    errors: list[str] = []
    warns: list[str] = []
    oks: list[str] = []

    if not zpath.exists():
        print(f"[ERROR] 文件不存在: {zpath}")
        return 1
    size = zpath.stat().st_size
    (oks if size <= SIZE_WARN else warns).append(f"包大小 {size:,} B")

    fp = fingerprint(zpath)
    if subs is None:
        warns.append("未提供 --submissions：重试守卫未激活（回路账本应在 submit 后维护）")
    else:
        prior = json.loads(subs.read_text(encoding="utf-8"))
        same = [s for s in prior if s.get("fingerprint") == fp]
        errored = [s for s in same if str(s.get("status", "")).lower() == "error"]
        succeeded = [s for s in same if str(s.get("status", "")).lower() == "complete"]
        if errored:
            errors.append(f"包指纹 {fp} 与曾 ERROR 的提交 {errored[0].get('ref')} 相同——改包再出门，不烧预算重试同包")
        elif succeeded:
            warns.append(f"包指纹 {fp} 与已成功提交 {succeeded[0].get('ref')} 相同——无增量")
        else:
            oks.append(f"包指纹 {fp} 未出现过")

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()

        data_files = [n for n in names if n.rsplit(".", 1)[-1].lower() in ("csv", "parquet")]
        if len(data_files) == 1:
            name = data_files[0]
            oks.append(f"唯一数据文件: {name}")
            if name.endswith(".csv"):
                header = z.read(name).decode("utf-8-sig", "replace").splitlines()[0]
                cols = [c.strip() for c in header.split(",")]
                if id_column in cols:
                    oks.append(f"csv 含 {id_column} 列（首列序: {cols[:3]}...）")
                else:
                    errors.append(f"csv 缺 {id_column} 列（平台将 ERROR 'ID column ... not found'）: {cols}")
                if expect_rows is None:
                    warns.append("未提供 --expect-rows：行数规则跳过（从 host solution 标定后传入）")
                else:
                    n_rows = sum(1 for _ in z.open(name)) - 1
                    if n_rows == expect_rows:
                        oks.append(f"csv 数据行数 {n_rows} == host solution 行数")
                    else:
                        errors.append(
                            f"csv 数据行数 {n_rows} != {expect_rows}（平台将 ERROR 'N rows but M were expected'）"
                        )
        elif len(data_files) > 1:
            errors.append(
                f"csv/parquet 多于一个（{len(data_files)} 个，平台将 ERROR 'multiple files'）: "
                + ", ".join(data_files[:4])
            )
        else:
            errors.append(
                "无任何 csv/parquet（平台将 ERROR 'No .csv or .parquet found'）——唯一数据文件由赛题 deliverables 决定"
            )

        for req in require:
            (oks if req in names else errors).append(f"必备交付物 {req}" + ("" if req in names else " 缺失"))

        junk = sorted({m for m in JUNK_MARKERS for n in names if m in n})
        (oks if not junk else warns).append("构建残渣: " + (", ".join(junk) if junk else "无"))

        for nb in (n for n in names if n.endswith(".ipynb")):
            try:
                cells = json.loads(z.read(nb)).get("cells", [])
                (oks if cells else errors).append(f"notebook {nb} ({len(cells)} cells)")
            except json.JSONDecodeError as e:
                errors.append(f"notebook {nb} 非法 JSON: {e}")

    for line in oks:
        print(f"[PASS] {line}")
    for line in warns:
        print(f"[WARN] {line}")
    for line in errors:
        print(f"[ERROR] {line}")
    print(f"\n指纹 {fp} | {len(oks)} PASS / {len(warns)} WARN / {len(errors)} ERROR")
    return 1 if errors else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip", type=Path, help="提交包路径")
    ap.add_argument("--submissions", type=Path, default=None, help="回路账本 submissions.json（重试守卫）")
    ap.add_argument("--id-column", default="id", help="id 列名（默认 id）")
    ap.add_argument("--expect-rows", type=int, default=None, help="host solution 行数（从赛题标定）")
    ap.add_argument("--require", default="", help="必备交付物相对路径清单，逗号/空格分隔（SPEC 阶段给定）")
    ap.add_argument("--print-fingerprint", action="store_true", help="只打印包指纹后退出（submit 记账用）")
    args = ap.parse_args()
    if args.print_fingerprint:
        print(fingerprint(args.zip))
        return
    require = [r for r in args.require.replace(",", " ").split() if r]
    raise SystemExit(lint(args.zip, args.submissions, args.id_column, args.expect_rows, require))


if __name__ == "__main__":
    main()
