#!/bin/sh
# wheel 版本提取共享件（红线 2 收敛）：Makefile WHL_VERSION 与 e2e 驱动共用，
# 事实单源 = pyproject.toml 的 version 字段。用法：_whl_version.sh [pyproject 路径]
sed -n 's/^version = "\(.*\)"/\1/p' "${1:-pyproject.toml}"
