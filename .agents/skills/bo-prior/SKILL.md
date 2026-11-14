---
name: bo-prior
description: BO 先验讨论问卷(mflowy)——产出 BOCard（bo 域先验卡 YAML）后交棒 bo-suggestion，不开局。用户要求"讨论实验先验/建 BO 卡/定搜索空间与业务规则"时触发；bo-suggestion 开局幕遇无卡时代调。继承上游 DomainCard（expected_ranges→搜索边界直取）与预测模型面，三幕收口落 <task_dir>/BOCard.yaml。
disable-model-invocation: true
---

# BO-Prior Skill：BO 先验卡问卷（bo-prior → BOCard → 交棒不开局）

你的职责：把 bo 推荐的对局先验问清楚、编译成 **BOCard**（bo 先验卡，pydantic 契约），
落 `<task_dir>/BOCard.yaml` 后**交棒 bo-suggestion 剧本继续开局**——本问卷不做开局
决策（survey→DomainCard 同构；不效仿 model-brief 直调 start 的先例）。

## 幕一：继承与列角色

**预填不问**（证据来自上游产物，逐项展示让用户确认/改）：

- 搜索边界 = DomainCard `expected_ranges` **直取**（物理域唯一权威在上游；缺列时卡
  自持该维边界——这本身就是缺口的表达，不回写上游）；卡声明宽于物理域会被校验拒绝；
- dtype / log scale 建议值、预测模型的 x_names、湿实验历史 schema 与统计；
- **知识节（通用方法论）**——三空间判据：特征空间（搜索维+边界）/ 目标空间
  （direction + 目标带）/ 置信度空间（v1 无载体——预测通道恒点值，上游区间/OOD
  就绪前靠典型测量变异 w）；变性/活性拆分：变性收窄走**按类常量表查表硬约束**，
  不进软约束（软约束唯一形态 = band 标量——构造性隔离，结构上不可能写错）；
  酶查表等域特异逻辑属案例知识，不进卡结构。

**问句（本幕唯一必问）**：列角色划分——哪些列是**优化变量**（进搜索空间：continuous
/ discrete / combinatorial 三 kind）、哪些是**钉值列（非搜索维）**（模型输入但不
优化——取值问用户：单类酶/固定协议等）、哪些是台账/目标列。判据提示：优化变量 =
实验室可调且值得探索的条件轴；钉值列 = 本轮固定但模型需要的输入。

## 幕二：策略组问齐（三档纪律：宁缺勿编）

- **必问**：direction（maximize/minimize/match + 目标带——match 必填带）、典型测量
  变异 w（目标单位，统计观测参考）、硬约束（表达式树：比较/跨特征算术/查表谓词 +
  按类常量表；复杂逻辑可给 python 逃生舱——`validate` 函数收单行候选 DataFrame
  返回 bool，白名单扫描）；
- **推荐确认**（你预填、用户改）：log scale（跨越数量级的连续维）、批大小缺省
  （实验室容量——必填项，可逐轮覆写）；
- **可空缺省**：两容差（接近/严重不符——无则 agent 主观判断 + 用户核实）、信号
  门限（缺省正向 100 / 负向 3 / 窗口门关——启用须 m < 负向门）。

## 幕三：收口编译

- BOCard 落 **`<task_dir>/BOCard.yaml`**（任务根，不进 study 子目录——当前卡唯一，
  study 指纹在 db attrs 审计）；YAML 写完逐项复述关键裁定让用户过目；
- **交棒提示**：转 bo-suggestion 剧本继续开局（签名 = 卡可空 + data|db 至少一 +
  model_run_id）；
- **纪律（提醒单落本剧本）**：改**身份字段**（搜索维 kind/边界/choices、direction）
  = study 重建——必提示走 bo-suggestion 的迁移式开局（`bo_load_or_init` 传 db +
  新空间组，在途推荐终局化、历史保态迁移）；策略组字段（规则/带/容差/门限/批缺省）
  可后改不重建。卡变更留痕 = 卡 YAML 文件本身（无 decision.log）。

## 参考

- BOCard 字段面与校验规则（match 必填带/规则树未知列拒绝/容差序/窗口门冗余）：
  [bo 域 README](../../../packages/builtin_plugins/bo/README.md)；
- 词面（BOCard/study/trial/推荐台账/湿实验/学习信号）：CONTEXT.md BO 节；
- 上游继承：DomainCard（survey 剧本产物）——expected_ranges/单位词表；
- 主剧本：bo-suggestion（owns 开局与全循环；卡对其只读）。
