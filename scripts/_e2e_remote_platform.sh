#!/bin/bash
# S0/S1 平台契约回归驱动（remote 实现系列 PR 合并门禁的平台面段——
# docs/research/2026-11-11-e2e-remote-runbook.md）：OTP 登录 → preset 断言 →
# 建实验 → presigned 上传 → CreateJob（cmd=sh 基座版，不经 wrapper）→ 轮询 →
# ENV/词表/modeling/zip 往返/三目录语义断言。改编自 #161 基座验收留档。
#
# 环境假设：本地 k3s dev 集群（gateway NodePort）、docker 可达 redis 容器、
# OTP 账号（未知邮箱静默建号）。**OTP 连续登录间隔 ≥65s**（per-IP 1/min +
# per-(purpose,email) 冷却）。钉住上游平台契约，与 mflowy 实现解耦——捕获
# 上游漂移（先例：mflowy-platform#45 regen 缺失、#46 scope 表）。
#
# 可覆盖参数（env）：
#   MFLOWY_BASE_URL               缺省 http://localhost:30080
#   MFLOWY_E2E_EMAIL              缺省 mflowy-base@mflowy.dev（OTP 验收账号）
#   MFLOWY_E2E_PRESET_ID          缺省种子 cpu-standard v2 id
#   MFLOWY_E2E_REDIS_CONTAINER    缺省 mflowy-infra-redis-1
#   MFLOWY_E2E_REDIS_PASSWORD_FILE 缺省 <monorepo 根>/.env.redis（逐级上溯）
#   MFLOWY_E2E_POLL_ITERATIONS    轮询次数（×6s，缺省 100）
set -uo pipefail

GW="${MFLOWY_BASE_URL:-http://localhost:30080}"
EMAIL="${MFLOWY_E2E_EMAIL:-mflowy-base@mflowy.dev}"
PRESET_ID="${MFLOWY_E2E_PRESET_ID:-c0000004-0000-0000-0000-000000000004}"  # cpu-standard v2 种子
REDIS_CONTAINER="${MFLOWY_E2E_REDIS_CONTAINER:-mflowy-infra-redis-1}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
jq_get() { python3 -c "import json,sys; d=json.load(sys.stdin); print(d$1)"; }
NPASS=0; NFAIL=0; NWARN=0
ok()   { echo "  ✓ $1"; NPASS=$((NPASS+1)); }
bad()  { echo "  ✗ $1"; NFAIL=$((NFAIL+1)); }
warn() { echo "  ⚠ $1"; NWARN=$((NWARN+1)); }
check(){ [ "$2" = "$3" ] && ok "$1" || bad "$1 — expected [$2] got [$3]"; }
die()  { echo "FATAL: $*" >&2; exit 2; }
# monorepo 标准检出布局：<platform>/apps/services/mflowy（git worktree 检出更深）；
# 逐级上溯找 .env.redis，非该布局时用 MFLOWY_E2E_REDIS_PASSWORD_FILE 显式指定
REDIS_PASSWORD_FILE="${MFLOWY_E2E_REDIS_PASSWORD_FILE:-}"
if [ -z "$REDIS_PASSWORD_FILE" ]; then
  d="$REPO_ROOT"
  for _ in 1 2 3 4 5; do
    d="$(dirname "$d")"
    if [ -f "$d/.env.redis" ]; then REDIS_PASSWORD_FILE="$d/.env.redis"; break; fi
  done
fi
[ -n "$REDIS_PASSWORD_FILE" ] || warn "未找到 .env.redis（无密码 redis 可继续；MFLOWY_E2E_REDIS_PASSWORD_FILE 可显式指定）"
# 版本断言单源 = pyproject（经 scripts/_whl_version.sh 与 Makefile WHL_VERSION 共用提取件）
WHL_VERSION="$(sh "$HERE/_whl_version.sh" "$REPO_ROOT/pyproject.toml")"
REDIS_PASSWORD="$(grep '^REDIS_PASSWORD=' "$REDIS_PASSWORD_FILE" 2>/dev/null | cut -d= -f2-)"
# 密码经 REDISCLI_AUTH 环境变量注入（redis-cli -a 会把密码暴露在 argv）
redis_cmd() {
  if [ -n "$REDIS_PASSWORD" ]; then
    docker exec -e REDISCLI_AUTH="$REDIS_PASSWORD" "$REDIS_CONTAINER" redis-cli --no-auth-warning "$@" 2>/dev/null
  else
    docker exec "$REDIS_CONTAINER" redis-cli --no-auth-warning "$@" 2>/dev/null
  fi
}
# 统一 API curl：静默 + 30s 超时（挂死的网关不应让轮询无限滞留）
api() { curl -s --max-time 30 "$@"; }

echo "== [0] OTP 登录（$EMAIL，未知邮箱静默建号；连续登录间隔须 ≥65s）=="
http=$(api -o /dev/null -w "%{http_code}" -X POST "$GW/api/v1/auth/send-verification-code/login" \
  -H 'Content-Type: application/json' -d "{\"email\":\"$EMAIL\"}")
[ "$http" = "200" ] || die "send-verification-code HTTP $http（429 限流？等 65s 重试）"
vc="$(redis_cmd get "vcode:login:$EMAIL" | tr -d '\r\n')"
# 数字校验 fail-loud：redis 未授权时 stdout 是错误文案而非验证码（NOAUTH 等），
# 静默送进 login 只会得到无 token 的错误体
case "$vc" in
  ''|*[!0-9]*) die "vcode 非数字（redis 读取失败——容器名/密码文件对？MFLOWY_E2E_REDIS_* 可覆盖；got: $vc）" ;;
esac
LOGIN=$(api -X POST "$GW/api/v1/auth/login" -H 'Content-Type: application/json' \
  -d "{\"email\":\"$EMAIL\",\"verification_code\":\"$vc\"}")
TOKEN=$(echo "$LOGIN" | jq_get "['token']" 2>/dev/null)
[ -n "$TOKEN" ] || die "login failed: $(echo "$LOGIN" | head -c 300)"
AUTH="Authorization: Bearer $TOKEN"
ok "web token 获取成功"

echo "== [0.5] preset 断言（S0.3：cpu-standard 在列且 workspace 1024Mi）=="
PR="$(api -w "\n%{http_code}" "$GW/api/v1/job-presets" -H "$AUTH")"
PHTTP="$(printf '%s\n' "$PR" | tail -1)"
if [ "$PHTTP" = "200" ]; then
  PBODY="$(printf '%s\n' "$PR" | sed '$d')"
  echo "$PBODY" | grep -q '"cpu-standard"' && ok "cpu-standard 在列" || bad "cpu-standard 不在列"
  echo "$PBODY" | python3 -c "
import json, sys
presets = json.load(sys.stdin)
p = next((x for x in (presets if isinstance(presets, list) else presets.get('presets', presets.get('items', [])))
          if x.get('name') == 'cpu-standard'), None)
sys.exit(0 if p and p.get('workspace_size_mi') == 1024 else 1)" \
    && ok "workspace 1024Mi" || warn "workspace_size_mi 非 1024（网关缺 #45 regen？记上游漂移）"
  echo "$PBODY" | grep -q "training-job:$WHL_VERSION" \
    && ok "preset image = training-job:$WHL_VERSION" \
    || warn "preset image 非 training-job:$WHL_VERSION（种子钉版前过渡轮：admin RPC 建版本钉 preset 对齐——#187）"
else
  warn "GET /api/v1/job-presets HTTP $PHTTP（web token 不可达——以 CreateJob 受理成功为替代证据，runbook S0.3）"
fi

echo "== [1] 建实验（全新 = workspace 断言无遗留）=="
RUN_ID="$(date +%s)-$RANDOM"
EID=$(api -X POST "$GW/api/v1/experiments" -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{\"name\":\"mflowy-base-$RUN_ID\"}" | jq_get "['experiment']['id']")
[ -n "$EID" ] || die "create experiment failed"
ok "experiment $EID"

echo "== [2] 上传 task.sh / diabetes.py / payload.json 到 uploads =="
for F in task.sh diabetes.py payload.json; do
  URL=$(api -X POST "$GW/api/v1/experiments/$EID/files/upload-url" -H "$AUTH" \
    -H 'Content-Type: application/json' -d "{\"mount_type\":\"uploads\",\"filename\":\"$F\"}" | jq_get "['upload_url']")
  [ -n "$URL" ] || die "no upload_url for $F"
  case "$F" in
    task.sh)     SRC="$HERE/_e2e_remote_task.sh" ;;
    payload.json) SRC="$HERE/_e2e_remote_payload.json" ;;
    diabetes.py)  SRC="$REPO_ROOT/examples/diabetes/diabetes.py" ;;
  esac
  CODE=$(api -o /dev/null -w "%{http_code}" -X PUT "$URL" --data-binary @"$SRC")
  check "upload $F" 200 "$CODE"
done

echo "== [3] CreateJob（cmd=sh 包装——geesefs 0666 无执行位；preset $PRESET_ID）=="
JR=$(api -X POST "$GW/api/v1/jobs" -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{\"cmd\":\"sh\",\"args\":[\"/user-data/uploads/task.sh\"],\"experiment_id\":\"$EID\",\"preset_id\":\"$PRESET_ID\"}")
JID=$(echo "$JR" | jq_get "['job']['id']")
[ -n "$JID" ] || die "CreateJob failed: $JR"
ok "job $JID（初始 $(echo "$JR" | jq_get "['job']['status']")）"

echo "== [4] 轮询至终态（modeling 全链，数分钟）=="
ST=""
for i in $(seq 1 "${MFLOWY_E2E_POLL_ITERATIONS:-100}"); do
  ST=$(api "$GW/api/v1/jobs/$JID" -H "$AUTH" | jq_get "['job']['status']")
  case "$ST" in succeeded|failed|lost|canceled) break ;; esac
  sleep 6
done
if [ "$ST" != "succeeded" ]; then
  # fail-loud 且可诊断：打印终态与日志尾，后续断言全部是级联噪声、不再逐条报
  echo "-- 终态 [$ST]（未达 succeeded；ImagePullBackOff = containerd 镜像缺，runbook S0.1 先导入）--"
  api "$GW/api/v1/jobs/$JID/logs?tail_lines=100" -H "$AUTH" | jq_get "['logs']" 2>/dev/null | tail -20
  die "终态非 succeeded: $ST（experiment=$EID job=$JID）"
fi
ok "终态 succeeded"

echo "== [5] 日志断言 =="
LOGS=$(api "$GW/api/v1/jobs/$JID/logs?tail_lines=500" -H "$AUTH" | jq_get "['logs']")
for m in "seam OK: task_dir = /user-data/workspace" "artifact_root = file:///user-data/workspace/.mlruns" \
         "modeling rc=0" "zip packed:" "uploads read-only OK" "task done job=" "mflowy $WHL_VERSION"; do
  echo "$LOGS" | grep -q "$m" && ok "logs 含 [$m]" || bad "logs 缺 [$m]"
done
echo "$LOGS" | grep -q "UNEXPECTED" && bad "uploads 意外可写" || ok "无 UNEXPECTED 标记"

echo "== [6] outputs 面：{JID}.zip 可见可下载、zip 内容契约 =="
OUT=$(api "$GW/api/v1/experiments/$EID/files?mount_type=outputs" -H "$AUTH")
echo "$OUT" | grep -q "$JID.zip" && ok "outputs/$JID.zip 可见" || bad "outputs/$JID.zip 不可见"
DL=$(api -X POST "$GW/api/v1/experiments/$EID/files/download-url" -H "$AUTH" \
  -H 'Content-Type: application/json' -d "{\"mount_type\":\"outputs\",\"filename\":\"$JID.zip\"}" | jq_get "['download_url']")
api "$DL" -o "$TMP/job.zip" && ok "zip 下载 ($(du -h "$TMP/job.zip" | cut -f1))" || bad "zip 下载失败"
unzip -l "$TMP/job.zip" | grep -q "mlflow.db" && ok "zip 含 mlflow.db" || bad "zip 缺 mlflow.db"
unzip -l "$TMP/job.zip" | grep -q ".mlruns/" && ok "zip 含 .mlruns/ 树" || bad "zip 缺 .mlruns/ 树"
# 最小链工件清单 = #161 实证三件（load/clean/split 各一；model 步工件需 test 数据边，
# 最小链不落——HPO 形态 zip 契约属 S5+）
for a in regressor_evidence.parquet split.pkl python_clean_report.json; do
  unzip -l "$TMP/job.zip" | grep -q "$a" && ok "zip 含 $a" || bad "zip 缺 $a"
done

echo "== [7] workspace 面：.scratch 写过但从未进 S3 → 404/403 =="
DL=$(api -X POST "$GW/api/v1/experiments/$EID/files/download-url" -H "$AUTH" \
  -H 'Content-Type: application/json' -d '{"mount_type":"workspace","filename":".scratch"}' | jq_get "['download_url']")
WH=$(api -o /dev/null -w "%{http_code}" "$DL")
case "$WH" in 403|404) ok "workspace/.scratch 不可下载（HTTP $WH）" ;; *) bad "workspace/.scratch HTTP $WH" ;; esac
WS=$(api "$GW/api/v1/experiments/$EID/files" -H "$AUTH" | python3 -c "import json,sys; print('\"workspace\"' in json.dumps(json.load(sys.stdin)))")
check "ListFiles 无 workspace 根" False "$WS"

echo ""
echo "== 汇总: PASS=$NPASS FAIL=$NFAIL WARN=$NWARN  experiment=$EID job=$JID =="
[ "$NFAIL" = 0 ]
