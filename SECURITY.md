# Security Policy

## Reporting a Vulnerability

We take security seriously. Please **do not open a public issue** for
vulnerability reports. Report privately through either channel:

1. **GitHub private vulnerability reporting** (recommended) —
   [Security Advisories](https://github.com/ifoodsci-ai/mflowy/security/advisories/new)
2. **Email** — `15587025323@163.com` (maintainer)

Include, when possible:

- Affected version(s) and environment
- A minimal reproduction (steps, configuration, payload)
- Expected vs. actual behavior

You should receive a response within 5 business days. Please keep details
confidential until the issue is fixed and announced.

## Scope

In scope:

- Remote code execution / injection through MCP tool arguments
  (`modeling_steps_yaml`, `python_loader` / `<py_path>:<func>` entry, HTTP loader URLs, etc.)
- Path traversal / arbitrary file access via tool path arguments
- Data exfiltration through tool outputs, MLflow integration, or telemetry
- Dependency supply-chain issues

## Telemetry Privacy

Telemetry data-collection practices (what is collected, where it goes, consent
and withdrawal) are documented in [PRIVACY.md](PRIVACY.md). Any privacy
regression in telemetry is treated as a security issue.
