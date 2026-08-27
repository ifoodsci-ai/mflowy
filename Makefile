.PHONY: install build build-whl test lint fmt precommit clean help e2e-info e2e-statistic e2e ui

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
	@echo "  make lint             运行 ruff 检查"
	@echo "  make fmt              运行 ruff 格式化"
	@echo "  make precommit        安装 pre-commit 钩子（commit 时自动 lint + UT）"
	@echo "  make clean            清理临时文件"
	@echo "  make e2e              运行 e2e 测试"
	@echo "  make ui [P=]          启动 MLflow UI 查看实验记录"
	@echo ""
	@echo "镜像覆盖（可选）:"
	@echo "  make build IMAGE_TAG=v1.0.0"
	@echo "  make build IMAGE_REGISTRY=localhost:5000"
	@echo "  make build MFLOWY_EXTRA_MODULES=\"mflowy-extra==0.1\"  定制插件包"

# 安装全部依赖
install:
	@[ -d .venv ] || uv venv
	uv sync --all-extras --all-groups
	uv lock

# 运行全量测试（tests/ 根 + packages/*/tests，见 pyproject testpaths）
test:
	uv run pytest

# 构建 wheel（workspace 五 distribution 锁步，uvx 分发产物 + 镜像构建输入）
build-whl:
	uv build --all --wheel -o dist/

# wheel 版本（pyproject.toml 单一来源，注入 Dockerfile ARG VERSION）
WHL_VERSION ?= $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)

# 构建 Docker 镜像（全部依赖）
IMAGE          = training-job:latest

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

e2e-info:
	uv run cmd list_modules
	uv run cmd list_modules '{"step": "model"}'
	uv run cmd get_module_info '{"step": "load", "module": "python"}'
	uv run cmd get_module_info '{"step": "clean", "module": "drop_missing"}'
	uv run cmd get_module_info '{"step": "X_y", "module": "x_y"}'
	uv run cmd get_module_info '{"step": "cross_validate", "module": "simple_cv"}'
	uv run cmd get_module_info '{"step": "x_transformer", "module": "standard_scaler"}'
	uv run cmd get_module_info '{"step": "model", "module": "XGB"}'
	uv run cmd get_module_info '{"step": "plot", "module": "taylor_diagram"}'
	uv run cmd validate_modeling_steps '{"modeling_steps_yaml": "$(abspath examples/diabetes/diabetes.yaml)"}'

e2e-statistic:
	uv run --extra stats cmd data_profile '{"file_path": "$(abspath examples/house_prices/house_prices.py:load)"}'
	uv run --extra stats cmd eda '{"file_path": "$(abspath examples/house_prices/house_prices.py:load)", "target": "SalePrice", "corr_method": "pearson", "top_k": 5, "cat_cols": ["HouseStyle", "Neighborhood"]}'
	uv run --extra stats cmd infer_task_type_by_statistic '{"file_path": "$(abspath examples/house_prices/house_prices.py:load)", "target": "SalePrice"}'

e2e: e2e-info e2e-statistic
	uv run --extra modeling cmd modeling '{"modeling_steps_yaml": "$(abspath examples/diabetes/diabetes.yaml)", "name": "糖尿病多模型对比", "desc": "standard_scaler → cv(8:1:1) → XGBoost/LightGBM/CatBoost/RF/MLP"}'
	@set -e; \
	out=$$(uv run python scripts/_e2e_gen.py); \
	exp_id=$$(printf '%s\n' "$$out" | sed -n 1p); \
	model=$$(printf '%s\n' "$$out" | sed -n 2p); \
	run_id=$${model#XGB=}; \
	echo "== mlflow 查询（experiment=$$exp_id, $$model）=="; \
	uv run cmd list_runs '{"experiment_id": "'$$exp_id'", "filter_steps": ["model.XGB"], "max_results": 3}'; \
	uv run cmd get_run '{"run_id": "'$$run_id'"}'; \
	uv run cmd list_run_artifacts '{"run_id": "'$$run_id'"}'; \
	echo "== SHAP 解释（$$model）=="; \
	uv run --extra modeling cmd explanation '{"modeling_steps_yaml": "$(abspath examples/diabetes/diabetes.yaml)", "model": "'$$model'", "name": "糖尿病 SHAP 解释", "desc": "XGB SHAP 全量样本"}'; \
	echo "== 预测（load_X 10 行采样）=="; \
	uv run --extra modeling cmd predict '{"data": "$(abspath examples/diabetes/diabetes.py:load_X)", "model": "'$$model'"}'; \
	echo "== 逆向设计（target minimize, 200 trials）=="; \
	uv run --extra modeling cmd inverse_optimization '{"data": "$(abspath examples/diabetes/diabetes.py:load_X)", "model": "'$$model'", "direction": {"target": "minimize"}, "n_trials": 200}'

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
