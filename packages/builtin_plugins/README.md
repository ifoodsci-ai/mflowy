# builtin_plugins/（内置域包分组）

两域包住此分组目录——目录名与 entry points 组 `mflowy.builtin_plugins` 机械对齐
（第三方插件声明进 `mflowy.plugins` 组，安装即注册）：

- [data — 数据域 catalog](data/README.md)：清洗 MDP 动作域（load / statistic / detect / clean + plot）
- [model — 建模域 catalog](model/README.md)：建模 MDP 动作域 + JobProvider 协议

第三方插件包以 mflowy 为 base 依赖，声明 `[project.entry-points."mflowy.plugins"]`
即成插件；**分组目录内只允许 distribution 目录**（uv workspace 的
`packages/builtin_plugins/*` glob 对无 pyproject 的子目录 fail-loud）。
