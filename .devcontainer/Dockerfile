FROM python:3.11-slim

ARG USERNAME
ARG USER_UID
ARG USER_GID
ARG WORKDIR

# 配置 apt 阿里云镜像源（加速软件下载）
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装系统依赖（含 Tauri v2 所需的 webkit/gtk 等库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    openssh-client \
    build-essential \
    libgomp1 \
    graphviz \
    # Noto Sans/Serif CJK 字体
    fonts-noto-cjk
    # 包含 Tinos (与 Times New Roman 兼容) [reference:0]
    fonts-croscore \
    # 字体配置工具 [reference:1]
    fontconfig \
    libwebkit2gtk-4.1-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libsoup-3.0-dev \
    libjavascriptcoregtk-4.1-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# 更新字体缓存，让系统识别新字体 [reference:2]
RUN fc-cache -fv
# Rust 工具链（从官方镜像复制到用户目录，免安装 rustup）
COPY --from=rust:1-slim-bookworm /usr/local/cargo /home/$USERNAME/.cargo
COPY --from=rust:1-slim-bookworm /usr/local/rustup /home/$USERNAME/.rustup

# Docker CLI + Compose + buildx 插件（DooD，需要挂载宿主机docker socket）
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker/compose-bin:latest /docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
COPY --from=docker/buildx-bin:latest /buildx /usr/local/lib/docker/cli-plugins/docker-buildx

# 添加系统用户并赋予 sudo 权限
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 预创建挂载点目录（防止 Docker 以 root 创建）
RUN mkdir -p /venv \
    && chown -R $USERNAME:$USERNAME /venv \
    && mkdir -p /home/$USERNAME/.local/bin \
    && mkdir -p /home/$USERNAME/.local/share/man \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

# 将虚拟环境创建在 /venv，彻底避开将要被挂载的 /app 目录
ENV MPLBACKEND=Agg \
    UV_PROJECT_ENVIRONMENT="/venv" \
    RUSTUP_HOME="/home/$USERNAME/.rustup" \
    CARGO_HOME="/home/$USERNAME/.cargo" \
    PATH="/venv/bin:/home/$USERNAME/.local/bin:/home/$USERNAME/.cargo/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER $USERNAME
WORKDIR /home/$USERNAME/$WORKDIR
CMD ["bash"]
