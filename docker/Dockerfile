# ========== 阶段 1：构建器 ==========
FROM python:3.12-slim-bookworm AS builder

# 拷贝 uv 二进制（来自 uv-source stage，不在 distroless uv 镜像里执行）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
ARG VERSION
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --all-extras --no-install-project
COPY dist/mflowy-${VERSION}-py3-none-any.whl /tmp/
# 项目本体从 wheel 安装（= 用户 uvx wheel 安装所得）；依赖已由 uv.lock 同步，--no-deps 跳过解析
RUN UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    uv pip install --python /app/.venv/bin/python --no-deps /tmp/mflowy-${VERSION}-py3-none-any.whl

# ========== 阶段 2：运行时 ==========
FROM python:3.12-slim-bookworm

# 系统运行时依赖
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    graphviz \
    # Noto Sans/Serif CJK 字体
    fonts-noto-cjk \
    # 包含 Tinos (与 Times New Roman 兼容) [reference:0]
    fonts-croscore \
    # 字体配置工具 [reference:1]
    fontconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# 更新字体缓存，让系统识别新字体 [reference:2]
RUN fc-cache -fv
WORKDIR /app
COPY --from=builder --chown=nobody:nogroup /app/.venv /app/.venv
USER nobody
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp
CMD ["cmd"]
