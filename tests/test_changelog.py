"""CHANGELOG 发布对账：版本头唯一 + 区块单调（#220 防再犯——deeec08c 合并事故曾整段重复粘贴）"""

import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"

_HEADER_RE = re.compile(r"^## \[(?P<version>[^\]]+)\]", re.MULTILINE)


def _version_headers() -> list[str]:
    return _HEADER_RE.findall(CHANGELOG.read_text(encoding="utf-8"))


def test_version_headers_unique():
    """同一版本号只允许一个 `## [x.y.z]` 头（重复粘贴事故的直接断言）"""
    headers = _version_headers()
    duplicates = {v for v in headers if headers.count(v) > 1}
    assert not duplicates, f"CHANGELOG 版本头重复（合并事故特征）: {sorted(duplicates)}"


def test_unreleased_first_and_descending():
    """[Unreleased] 居首，后续版本头降序排列（发布切版时防错位插入）"""
    headers = _version_headers()
    assert headers[0] == "Unreleased", f"[Unreleased] 须居首，实际首个: {headers[0]}"
    versions = headers[1:]

    def _key(v: str) -> tuple[int, ...]:
        parts = v.split(".")
        assert all(p.isdigit() for p in parts), f"非语义化版本头: {v}"
        return tuple(int(p) for p in parts)

    assert versions == sorted(versions, key=_key, reverse=True), f"版本头须降序: {versions}"
