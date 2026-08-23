# Contributing to MFlowy

Thanks for your interest! MFlowy is an MIT-licensed, MCP-native ML workflow
engine. Contributions of all kinds are welcome — features, bug fixes,
documentation, examples, and issue triage.

## Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).
Please also read [PRIVACY.md](PRIVACY.md) — the telemetry privacy contract —
before touching anything telemetry-related. Report suspected vulnerabilities
privately via the channels in [SECURITY.md](SECURITY.md).

## Getting Started

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ifoodsci-ai/mflowy.git
cd mflowy
uv sync --all-extras --all-groups   # all deps (stats + modeling + dev)
make test                           # full test suite
make lint                           # ruff check
make fmt                            # ruff format
```

## Finding Work

- Open issues in the [issue tracker](https://github.com/ifoodsci-ai/mflowy/issues)
  use canonical labels: `needs-triage`, `needs-info`, `ready-for-agent`,
  `ready-for-human`, `wontfix` (see [AGENTS.md](AGENTS.md) — Triage Labels).
- Look for issues tagged `ready-for-agent` or `good first issue`.
- To avoid duplicate work, comment on the issue before starting.

## Development Workflow

1. **Branch** — `git checkout -b feat/my-change`.
2. **Code** — follow the project's coding principles
   ([AGENTS.md](AGENTS.md) — Design Principles):
   - Python 3.12, line length 120, ruff lint (`E4/E7/E9/F/I/UP035/UP037`).
   - User-facing configurable params are annotated `Annotated[T, "描述"]`.
   - New compute entities register via the `@handler` decorator — no manual wiring.
   - Keep new dependencies out of `base` unless truly needed; data-stack
     dependencies go in `[stats]` / `[modeling]` extras.
3. **Test** — add tests mirroring the code path (`tests/` mirrors `mflowy/`).
   Run `make test` before pushing.
4. **Lint** — `make lint && make fmt`.
5. **Docs** — tool behavior changes update the **docstring** (it compiles into
   the MCP schema users actually see; see [AGENTS.md](AGENTS.md) —
   Documentation Model).
6. **Commit** — Conventional Commits style with a Chinese description:

   ```
   feat(mcp): 新增 xxx 工具
   fix(driver): 修复 xxx 解析问题
   docs(examples): 补充 xxx 案例
   ```

7. **PR** — open against `main`, reference the issue if applicable.
   CI (ruff + pytest) must pass.

## Project Navigation

- [README.md](README.md) — overview, quick start, architecture
- [AGENTS.md](AGENTS.md) — architecture, commands, layer model, docs model, agent conventions
- [docs/research-flow.md](docs/research-flow.md) — methodology guide for users
- [docs/roadmap.md](docs/roadmap.md) — feature roadmap
- [docs/DRIVER.md](docs/DRIVER.md) — DAG kernel: design philosophy & architecture (`src/mflowy/driver/`)
- [docs/REMOTE_MODELING.md](docs/REMOTE_MODELING.md) — remote execution (JobProvider contract)
- [docs/TELEMETRY.md](docs/TELEMETRY.md) — telemetry setup and configuration

## Telemetry

MCP tool-call telemetry is **consent-based** (default `ask` via MCP elicitation;
see [PRIVACY.md](PRIVACY.md)). Instrumentation lives at the protocol layer —
agentcat wraps the `MCPServer` in `src/mflowy/mcp/server.py` via
`src/mflowy/mcp/telemetry.py:wire_agentcat` — so **do not** add per-tool wrappers
or emit telemetry from `src/mflowy/mcp/tools.py`. Any change to what is collected,
where it goes, or how consent works must update [PRIVACY.md](PRIVACY.md) in
the same PR.
