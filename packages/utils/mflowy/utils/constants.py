"""全局常量"""

# 随机种子，用于 CV/模型/SHAP 等所有依赖可重现随机性的模块
RANDOM_STATE: int = 42

# 远程任务镜像契约字面常量（#159 §4 / #218 单源收敛）：**非 env 读取**——与远端
# 镜像烧 ENV 契约无关（镜像面见 docker/Dockerfile 烧 ENV、容器面见 jobwrap）。
# workspace = 容器内任务工作区根（三挂载之一；zip 往返根）；uploads = 上传清单
# 远端寻址前缀（basename 平铺）。消费方：app 观察面（工件路径改写/同步 diff 条目）
# 与 model 会话 remote 分支（plan 渲染 uploads 寻址 / 解包工件 URI 改写）。
REMOTE_TASK_DIR = "/user-data/workspace"
REMOTE_UPLOADS_DIR = "/user-data/uploads"
