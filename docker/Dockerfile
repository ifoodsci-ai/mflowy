# ========== 阶段 1：构建器 ==========
FROM python:3.12-slim-bookworm AS builder

# 拷贝 uv 二进制（来自 uv-source stage，不在 distroless uv 镜像里执行）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
ARG VERSION
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --all-extras --no-install-project
COPY dist/ /tmp/wheels/
# 项目本体从 wheel 安装（= 用户 uvx wheel 安装所得）；依赖已由 uv.lock 同步，--no-deps 跳过解析；
# 五 wheel（mflowy 聚合 + 四子包）顺序无关
RUN UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    uv pip install --python /app/.venv/bin/python --no-deps /tmp/wheels/*.whl

# 可选定制：额外插件包（空格分隔，如 "mflowy-extra==0.1"）。
# 装进镜像后其 mflowy.plugins entry points 随 discover 自动生效，运行期零配置
ARG MFLOWY_EXTRA_MODULES=""
RUN if [ -n "$MFLOWY_EXTRA_MODULES" ]; then \
    uv pip install --python /app/.venv/bin/python $MFLOWY_EXTRA_MODULES; fi

# ========== 阶段 2：运行时 ==========
FROM python:3.12-slim-bookworm

# 系统运行时依赖
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    graphviz \
    # 远程任务 zip 往返（产物打包/继承解包，任务包装器与调用方参考模式共用）
    zip \
    unzip \
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
# HOME 必须可写（镜像契约，#161 研究回填）：nobody 缺省 HOME 不可写，而
# mflowy/utils/mlflow.py setup() 急切 mkdir $HOME/.mflowy、mlflow 遥测自写
# $HOME/.config/mlflow——指向 /tmp（per-pod 临时，不占 workspace 配额、不进 zip）。
# 远程任务面（平台 Job 容器入口）：三挂载烧 ENV——task_dir 即 workspace
# （emptyDir scratch），mlflow 走库内 sqlite（mlflow.db 与 .mlruns/ 同根，
# zip 往返契约可推导）。本地/MCP 部署不受影响：ENV 缺省值语义不变，显式
# 覆盖（run -e）优先。workspace 为 emptyDir 0777，nobody 可写（k3s 实测）。
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp \
    HOME=/tmp \
    MFLOWY_TASK_DIR=/user-data/workspace \
    MLFLOW_TRACKING_URI=sqlite:////user-data/workspace/mlflow.db
CMD ["mflowy"]
