.PHONY: install build build-whl test test-cov typecheck lint fmt precommit clean help e2e-info e2e-statistic e2e-mcp e2e-bench e2e-remote e2e-remote-tools e2e ui

# 默认目标
help:
	@echo "MFlowy Makefile"
	@echo ""
	@echo "镜像: $(IMAGE)"
	@echo ""
	@echo "可用命令:"
	@echo "  make install          安装全部依赖"
	@echo "  make build            构建 Docker 镜像（全部依赖，自动先建 wheel）"
	@echo "  make build-whl        构建 wheel 至 dist/（uvx 分发与镜像构建产物）"
	@echo "  make test             运行测试"
	@echo "  make test-cov         运行测试并产出覆盖率（term 日志 + coverage.xml，CI 消费）"
	@echo "  make typecheck        运行 pyright 类型检查（basic 档，门范围 packages+tests）"
	@echo "  make lint             运行 ruff 检查"
	@echo "  make fmt              运行 ruff 格式化"
	@echo "  make precommit        安装 pre-commit 钩子（commit 时自动 lint + UT）"
	@echo "  make clean            清理临时文件"
	@echo "  make e2e              三入口分发冒烟（CLI/stdio 真实进程 + diabetes 全链，发布前人肉跑）"
	@echo "  make e2e-bench        六基准冒烟门（iris/digits/titanic + 工业三链 + 分类会话链 + 应用面三工具，判据含指标下限）"
	@echo "  make e2e-remote       S0/S1 平台契约回归（remote 系列 PR 门禁，需本地 k3s dev 集群）"
	@echo "  make e2e-remote-tools  S2+ 工具面门禁（MCP stdio 通道，需已登录 + k3s 集群）"
	@echo "  make ui [P=]          启动 MLflow UI 查看实验记录"
	@echo ""
	@echo "镜像覆盖（可选，tag 缺省 = 版本钉 training-job:\$$(WHL_VERSION)，latest 已退役）:"
	@echo "  make build IMAGE=\"<registry>/training-job:<版本>\"  整串覆盖（如生产 push images.mflowy.com）"
	@echo "  make build MFLOWY_EXTRA_MODULES=\"mflowy-extra==0.1\"  定制插件包"

# 安装全部依赖
install:
	@[ -d .venv ] || uv venv
	uv sync --reinstall-package mflowy-data --all-extras --all-groups
	uv lock

# 运行全量测试（tests/ 根 + packages/*/tests，见 pyproject testpaths）
test:
	uv run pytest

# 覆盖率面（#43）：数字不设阈值——term 报告进日志，coverage.xml 供 CI 工件/后续阈值化
test-cov:
	uv run pytest --cov=mflowy --cov-report=term --cov-report=xml

# 类型门（#42 决议→#91）：pyright basic 全仓门（packages+tests）；配置单源在根
# pyproject [tool.pyright]（venvPath+venv 键——pythonPath 键静默失效）
typecheck:
	uv run pyright

# 构建 wheel（workspace 五 distribution 锁步，uvx 分发产物 + 镜像构建输入）
build-whl:
	uv build --all --wheel -o dist/

# wheel 版本（pyproject.toml 单一来源，注入 Dockerfile ARG VERSION；提取件与 e2e 驱动共享）
WHL_VERSION ?= $(shell sh scripts/_whl_version.sh pyproject.toml)

# 构建 Docker 镜像（全部依赖）。tag = pyproject 版本单源（#187 统一版本钉：
# latest 退役——dev 同 tag 重建 = 测试位、链中途禁重导；生产 push 不可变 tag）；
# 整串覆盖走命令行 IMAGE=（如 images.mflowy.com/training-job:<V>）
IMAGE          = training-job:$(WHL_VERSION)

# 额外插件包（空格分隔，如 make build MFLOWY_EXTRA_MODULES="mflowy-extra==0.1"）
MFLOWY_EXTRA_MODULES ?=

build: test build-whl
	docker build \
		--build-arg VERSION=$(WHL_VERSION) \
		--build-arg MFLOWY_EXTRA_MODULES="$(MFLOWY_EXTRA_MODULES)" \
		-t $(IMAGE) -f docker/Dockerfile .

# 运行 ruff 检查
lint:
	uv run ruff check .
	uv run ruff format --check

# 运行 ruff 格式化
fmt:
	uv run ruff format .

# 安装 pre-commit 钩子
precommit:
	uv run pre-commit install

# e2e 工具调用的 modeling_steps 明文载荷（#66 收口：工具收 YAML 明文，文件路径参数已退役）
DIABETES_STEPS = $(abspath examples/diabetes/diabetes.yaml)
PAYLOAD = uv run python scripts/_e2e_payload.py

# e2e 追踪隔离（一次性冒烟不写机器级默认库 ~/.mflowy/mlflow.db——P2-7 污染教训；
# recipe 以 shell 默认值形式注入，已设 MLFLOW_TRACKING_URI 则尊重）。:= 单次展开
# （惰性 ?= 会让 $(shell mktemp) 每条 recipe 重新求值 → modeling 与查询各命中各库）
ifeq ($(origin E2E_TRACKING), undefined)
E2E_TRACKING := sqlite:///$(shell mktemp -u /tmp/mflowy_e2e_XXXXXX).db
endif

e2e-info:
	uv run mflowy list_nodes
	uv run mflowy list_nodes '{"namespace": "model"}'
	uv run mflowy file_hash '{"path": "$(DIABETES_STEPS)"}'
	uv run mflowy get_node_info '{"namespace": "load", "node_name": "python"}'
	uv run mflowy get_node_info '{"namespace": "clean", "node_name": "drop_missing_rows"}'
	uv run mflowy get_node_info '{"namespace": "", "node_name": "x_y"}'
	uv run mflowy get_node_info '{"namespace": "cv", "node_name": "split"}'
	uv run mflowy get_node_info '{"namespace": "x_transformer", "node_name": "standard_scaler"}'
	uv run mflowy get_node_info '{"namespace": "model", "node_name": "XGB"}'
	uv run mflowy get_node_info '{"namespace": "plot", "node_name": "taylor_diagram"}'
	uv run mflowy validate_modeling_steps "$$($(PAYLOAD) "$(DIABETES_STEPS)")"
# 分析类工具冒烟（本地执行，不经 JobProvider）：数据画像 / EDA。
# 数据集：运行时生成的小型分类数据（mktemp 临时 csv，退场即删）——e2e 不依赖外部数据文件。
e2e-statistic:
	@set -e; \
	csv=$$(mktemp /tmp/mflowy_e2e_XXXXXX.csv); \
	trap 'rm -f "$$csv"' EXIT; \
	uv run python -c "import numpy as np, pandas as pd; rng = np.random.default_rng(7); n = 120; df = pd.DataFrame({f'f{i}': rng.normal(size=n) for i in range(4)}); df['target'] = rng.integers(0, 2, n); df.to_csv('$$csv', index=False)"; \
	MLFLOW_TRACKING_URI="$${MLFLOW_TRACKING_URI:-$(E2E_TRACKING)}" uv run mflowy data_profile '{"file_path": "'$$csv'"}'; \
	MLFLOW_TRACKING_URI="$${MLFLOW_TRACKING_URI:-$(E2E_TRACKING)}" uv run mflowy eda '{"file_path": "'$$csv'", "target": "target"}'

# MCP stdio 入口冒烟（三入口之一）：握手 → tools/list → tools/call
e2e-mcp:
	uv run python scripts/_e2e_mcp.py

# 六基准冒烟门（#75 三基准 + 工业三链）：iris 五族 / digits MLP+XGB / titanic 清洗链
# 三树族 + bearing_rul RUL 回归 / steel_defect 不平衡分类 / soft_sensor 传感器缺失清洗
# + 分类会话最小链 + 应用面三工具（#167：终训 run_id 寻址 explanation/predict/inverse
# ——工件存在性判据，无指标下限）。发布前人肉门 = 先 make lint && make typecheck 再 make e2e
# （三入口分发）再 make e2e-bench（#107 教训：切版对账漏 typecheck，债务直通 release PR）；
# 判据含指标下限（防绿但退化），越限 FAIL 行 + 非零退出。
# 越限自证：BENCH_FLOOR_BUMP='iris|RF|99' make e2e-bench
e2e-bench:
	uv run --extra model --extra deep python scripts/_e2e_bench.py

# S0/S1 平台契约回归（remote 实现系列 PR 合并门禁——docs/research/2026-11-11-e2e-remote-runbook.md）：
# 平台面 curl 驱动（OTP 登录 → 建实验 → 上传 → CreateJob cmd=sh → 轮询 → 断言），
# 钉住上游契约、与 mflowy 实现解耦。需本地 k3s dev 集群 + OTP 账号 + redis 容器，
# 人肉门（对齐 e2e/e2e-bench 形态，不进常规 PR 门）；OTP 连续登录间隔 ≥65s。
# 镜像前置（S0.1）：make build（版本钉 training-job:$(WHL_VERSION)）→ docker save → k3s 导入
e2e-remote:
	bash scripts/_e2e_remote_platform.sh

# S2+ 工具面 E2E（remote 系列门禁的工具段）：MCP stdio 真实通道，cwd 置临时目录。
# 需本地 k3s dev 集群 + ~/.mflowy/credentials.json 已登录（access 过期自动 refresh）；
# 断言面随切片生长（S2=ensure / S3+=提交·观察·拉取），人肉门
e2e-remote-tools:
	uv run python scripts/_e2e_remote_tools.py

e2e: e2e-info e2e-statistic e2e-mcp
	@set -e; \
	stage=$$(mktemp -d /tmp/mflowy_e2e_stage_XXXXXX); \
	trap 'rm -rf "$$stage"' EXIT; \
	export MLFLOW_TRACKING_URI="$${MLFLOW_TRACKING_URI:-$(E2E_TRACKING)}"; \
	uv run --extra model --extra deep mflowy modeling "$$($(PAYLOAD) "$(DIABETES_STEPS)" --stage "$$stage" 'name=糖尿病多模型对比' 'desc=standard_scaler → cv(8:1:1) → XGBoost/LightGBM/CatBoost/RF/MLP')"; \
	out=$$(uv run python scripts/_e2e_gen.py); \
	exp_id=$$(printf '%s\n' "$$out" | sed -n 1p); \
	run_id=$$(printf '%s\n' "$$out" | sed -n 2p | cut -d= -f2); \
	echo "== mlflow 查询（experiment=$$exp_id, MLP=$$run_id）=="; \
	uv run mflowy list_runs '{"experiment_id": "'$$exp_id'", "filter_steps": ["model.MLP"], "max_results": 3}'; \
	uv run mflowy get_run '{"run_id": "'$$run_id'"}'; \
	uv run mflowy list_run_artifacts '{"run_id": "'$$run_id'"}'; \
	echo "== SHAP 解释（MLP=$$run_id）=="; \
	uv run --extra model --extra deep mflowy explanation "$$($(PAYLOAD) "$(DIABETES_STEPS)" --stage "$$stage" "model=$$run_id" 'name=糖尿病 SHAP 解释' 'desc=MLP SHAP 全量样本')"; \
	echo "== 预测（全量 X 442 行）=="; \
	uv run --extra model --extra deep mflowy predict '{"data": "'$$stage/diabetes.py:load_X'", "model": "'$$run_id'"}'; \
	echo "== 逆向设计（target minimize, 200 trials）=="; \
	uv run --extra model --extra deep mflowy inverse_optimization '{"data": "'$$stage/diabetes.py:load_X'", "model": "'$$run_id'", "direction": {"target": "minimize"}, "n_trials": 200}'

# 启动 MLflow UI
DB_PATH := $(HOME)/.mflowy/mlflow.db
DB_URI ?= sqlite:///$(DB_PATH)
ui:
	uv run mlflow ui --host 0.0.0.0 --backend-store-uri $(DB_URI)

telemetry:
	@chmod +x docker/telemetry/generate-secret.sh
	@echo "TALOS_ENCRYPTION_KEY=$$(openssl rand -base64 32 | tr -d '\n' | cut -c1-32)" > docker/telemetry/.env
	@docker compose -f docker/telemetry/docker-compose.yml \
		--env-file docker/telemetry/.env \
		up -d --build

# 清理
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name mlflow.db -delete
	find . -type d -name .mlruns -exec rm -rf {} + 2>/dev/null || true
