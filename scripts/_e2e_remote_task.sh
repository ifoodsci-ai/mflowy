#!/bin/sh
# 远程任务基座载荷（#161 实证形态，runbook S1 保留基座版作平台回归——
# docs/research/2026-11-11-e2e-remote-runbook.md 附录）：ENV 回显 / 词表 /
# task_dir 锚 / 真实入口 modeling / 手工 zip 往返 / 三目录语义。
# 切片④落地后任务包装器取代 [5][6] 段，本脚本维持基座版不动。
# 刻意不 set -e、失败段不早退：各段输出即断言证据（驱动 grep "modeling rc=" 判成败），
# 容器契约（zip/ENV/只读探针）在 modeling 失败时仍可验证——job succeeded ≠ modeling 成功。
set -u

echo "== [1] 环境契约（镜像烧 ENV）=="
echo "MFLOWY_TASK_DIR=$MFLOWY_TASK_DIR"
echo "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
echo "MFLOWY_JOB_ID=$MFLOWY_JOB_ID"
echo "HOME=$HOME"

echo "== [2] 工具面 =="
command -v zip unzip python mflowy

echo "== [3] CLI 词表 =="
python -c "from importlib.metadata import version; print('mflowy', version('mflowy'))"
mflowy list_nodes | head -c 120; echo

echo "== [4] task_dir 锚（ContextVar：pin 后 seam 可用——双路由契约，runner 入口自动钉）=="
python - <<'PY'
import os
from pathlib import Path
Path(os.environ["HOME"], ".mflowy").mkdir(exist_ok=True)  # HOME 可写回填验证
from mflowy.utils.path import pin_task_dir, task_dir
pin_task_dir(os.environ["MFLOWY_TASK_DIR"])
from mflowy.utils.mlflow import MlflowTracking
import mlflow
t = MlflowTracking()
exp = t.ensure_experiment("base-verify", description="#161 remote base")
with t.node_run(os.environ.get("MFLOWY_JOB_ID", "probe"), exp.experiment_id):
    mlflow.log_param("probe", "remote-base")
print("seam OK: task_dir =", task_dir(), "| artifact_root =", exp.artifact_location)
PY

echo "== [5] 真实入口：mflowy modeling（CLI runner 钉 MFLOWY_TASK_DIR → workspace 闭环）=="
mflowy modeling "$(cat /user-data/uploads/payload.json)"
echo "modeling rc=$?"

echo "== [6] zip 往返打包（task_dir 全量 → outputs，S3 持久）=="
echo "workspace-only scratch" > /user-data/workspace/.scratch
cd /user-data/workspace && zip -qr "/user-data/outputs/${MFLOWY_JOB_ID}.zip" . && echo "zip packed: ${MFLOWY_JOB_ID}.zip"

echo "== [7] uploads 只读负向探针 =="
if echo x > /user-data/uploads/.write-probe 2>/dev/null; then
  echo "UNEXPECTED: uploads writable"
else
  echo "uploads read-only OK"
fi
echo "task done job=$MFLOWY_JOB_ID"
