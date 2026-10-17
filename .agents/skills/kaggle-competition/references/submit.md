# SUBMIT — 稀缺动作（判据见骨架）

## 有序调用链

1. `search_competition_submissions(competitionName, sortBy=Date)` → 数当日已提交次数，确认预算余量（**预算门在这里观测**）
2. `start_competition_submission_upload(competitionName, fileName, contentLength, lastModifiedEpochSeconds)` → 拿 blob token + 上传 URI
3. 原始 HTTP PUT 上传（**非 MCP 环节**）：URI 主机 `www.googleapis.com` 不可达时换 `storage.googleapis.com`（同一 GCS 后端，URI 其余不动）
4. `submit_to_competition(competitionName, blobFileTokens, submissionDescription)` → 拿提交 ref
5. 记账：submissions.json ⊕= (ref, ts, bytes, **包指纹**——用 lint 脚本 `--print-fingerprint` 取)

## 双段转移（提交语义核心）

- **即时段**：`submit_to_competition` 成功 = 预算 −1 + ref 到手；**技术失败（无 ref）= 预算未扣**，回 PACKAGE 排障后重走
- **延迟段**：校验结果**不在提交响应里**——ERROR 或分数由 poll 揭示 → 拿到 ref 后立即进入 POLL

## 陷阱

- blob token 抄写**逐字符核对**（一次手滑即 "Invalid token"）
- 提交描述写硬数字（评审第一眼看到的就是它）
- 校验失败回 PACKAGE 最小改包，**不烧新预算重试同一包**（指纹守卫已在 lint 拦截）

## 出边

ref 到手 → POLL（读 references/poll-revise.md）；技术失败 → PACKAGE 排障
