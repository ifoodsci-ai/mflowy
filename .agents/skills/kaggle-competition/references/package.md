# PACKAGE — 打包与 preflight（判据见骨架）

## 动作

1. 按 SPEC 的 deliverables 清单组装提交包（zip）
2. 跑活 lint（本 skill 目录）：

```bash
python scripts/submission_lint.py <zip> --submissions submissions.json \
  [--id-column id] [--expect-rows N] [--require f1,f2,...]
```

   - `submissions.json` = 回路账本（工作目录维护）：`[{"fingerprint", "ref", "status", "ts"}, ...]`；submit 记账时用 `--print-fingerprint` 取指纹
   - **同指纹且曾 ERROR → 拒绝出门**（不重试同包守卫，沉在代码不在纪律）；同指纹曾成功 → WARN 无增量
3. **lint 不过不出门**（build 脚本收尾应自动跑一次）

## 已知平台约束（实证沉淀；新 error_description 回流补一行 + 脚本补实现——飞轮沉积处）

| 规则 | 级别 |
| --- | --- |
| zip 内 .csv/.parquet **恰好一个**（有分赛=官方要求的提交文件；人工评审赛选数据产品作唯一数据文件，其余数据靠 notebook 再生并在 README 注明） | ERROR |
| 唯一 csv 含字面 id 列（`--id-column`）且数据行数 = host solution 行数（`--expect-rows`，从赛题标定） | ERROR |
| deliverables 必备件齐（`--require` 传 SPEC 清单：notebook/报告/验证脚本/PROVENANCE/README 按题） | ERROR |
| notebook 合法 JSON 且非空 | ERROR |
| 无构建残渣（`__pycache__`/`.mlruns`/`.DS_Store`）、包体 sane（<100MB） | WARN |

## 陷阱

- 定性赛挂占位 solution 时：唯一 csv 让位给小 manifest（如 3 行 id+target），真实数据产品转 xlsx 随包、csv 再生
- lint 只编码已发现的约束——平台校验器是黑箱，未发现的只能经提交反馈暴露（每条 ERROR 是一次约束发现）

## 出边

lint 0 ERROR ∧ 预算>0 ∧ 规则门过 ∧ 指纹未失败过 → SUBMIT（读 references/submit.md）
