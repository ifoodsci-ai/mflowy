FROM python:3.12-slim-bookworm

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=1000
ARG WORKDIR
ARG APT_MIRROR=""

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# apt 镜像源：空（默认）= 保持 Debian 官方源；非空 = 替换为指定镜像（如 compose 传入的 mirrors.aliyun.com）
RUN if [ -n "$APT_MIRROR" ]; then \
        sed -i "s/deb.debian.org/${APT_MIRROR}/g" /etc/apt/sources.list.d/debian.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    openssh-client \
    build-essential \
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

# Docker CLI + Compose + buildx 插件（DooD，需要挂载宿主机docker socket）
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker/compose-bin:latest /docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
COPY --from=docker/buildx-bin:latest /buildx /usr/local/lib/docker/cli-plugins/docker-buildx

# 添加系统用户并赋予 sudo 权限
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 预创建挂载点目录（防止 Docker 以 root 创建，破坏 claude 等宿主机挂载的所有权）
RUN mkdir -p /home/$USERNAME/.local/bin \
    && mkdir -p /home/$USERNAME/.local/share/man \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

# claude 等宿主机二进制经 ~/.local/bin 挂载进容器
ENV MPLBACKEND=Agg \
    PATH="/home/$USERNAME/.local/bin:$PATH"

USER $USERNAME
WORKDIR /home/$USERNAME/$WORKDIR
CMD ["bash"]
