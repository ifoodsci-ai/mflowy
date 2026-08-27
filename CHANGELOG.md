# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **插件化架构：词表从 StepType 枚举改为 entry points 目录**（BREAKING）：
  - 删除 `driver/config.py: StepType`，`StepConf.type` 改为 `str`，YAML 值不变（load/clean/X_y/...）
  - `@handler` 删除首参 step（签名变为 `@handler(*middlewares)`），身份由 entry point name `step.module` 声明；`handler.py` 注册表 `_REGISTRY`/`_POST_INIT_REGISTRY` 删除，改为函数属性挂载（`.handler` 调度链 + `.convert_params` 转换器）
  - `discover.py` 重写：读取 `mflowy.builtin_plugins`（内置，`hatch_metadata.py` 构建期 AST 扫描生成）与 `mflowy.plugins`（第三方，安装即注册，后组覆盖前组）两组 entry points；目录查询零 import，加载惰性且坏声明 fail-loud——旧扫描机制“缺 extra 静默丢模块”问题随之解决
  - 第三方插件：以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]` 即可；镜像支持 `make build MFLOWY_EXTRA_MODULES="pkg==ver"` 定制插件包
  - 注意：editable 安装下新增 compute 模块需重跑 `uv sync` 刷新 entry points 元数据

### Added

- **file_hash MCP 工具**: 文件指纹（sha256/md5/sha1，分块流式），供 agent 核验
  数据文件在阶段间未被改动

### Fixed

- **devcontainer**: 修复 Dockerfile 续行符缺失导致的解析失败与 `$APT_MIRROR`
  单引号不展开问题；`APT_MIRROR` 空默认（官方源）仅非空时替换，镜像源改由
  compose 显式传入；基础镜像对齐 3.12；移除仓库已不存在的 Tauri/Rust 依赖；
  恢复 `--no-install-recommends` 与 `~/.local/bin` 挂载点；`postCreateCommand`
  修正为 `make install`

## [0.1.2] - 2026-08-23

首个公开版本。（0.1.0 / 0.1.1 版本号已被 PyPI 文件名保留规则永久占用——曾上传的文件删除后文件名不可复用，故首发从 0.1.2 计。）

### Added

- **MCP-native 架构**: 全部能力以 pyfunc 工具暴露，MCP server（stdio）/ JSON
  runner（`cmd`）/ 直接 import 三入口；入口脚本定名 `cmd` / `mcpSrv`（避开
  mcp SDK 同名 CLI）；传输 stdio、身份经请求 `_meta` 透传
- **13 个 MCP 工具**: info（list_modules / get_module_info / validate_modeling_steps）、
  mlflow（list_runs / get_run / list_run_artifacts）、
  分析（data_profile / eda / infer_task_type_by_statistic，本地执行）、
  建模（modeling / explanation / predict / inverse_optimization，JobProvider 委派）
- **DAG 工作流引擎**: YAML 配置驱动（Jinja2 模板），`@handler` 装饰器自动注册，
  中间件链（数据注入 / mlflow 记录 / 错误即停），LIFO 深度优先拓扑调度，
  结构化 `WorkflowResult` 直传
- **依赖分层**: base / [stats] / [modeling] 三层 extras

### Changed

- **README 定位**: 自主数据分析/训练/预测 agent 的能力底座，Agent 自主回路
  （发现/组装/执行/读回）
- **协议卫生**: Taylor 图 vendor Copin taylorDiagram.py（public domain /
  CC BY 4.0），无 GPLv3 衍生代码
- **src 布局**: `src/mflowy/` 标准 src 布局，import 名 `mflowy`；
  Trusted Publisher CI 发布
