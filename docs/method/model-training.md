# 模型训练方法论参考（决策点依据）

> **定位**：研究领域知识——模型选族、超参搜索、特征工程、评估判读四类决策点的方法论依据，供建模会话下钻引用（某模型族的小样本行为、某编码的泄漏纪律、差距小于折间波动时怎么判读）。叙事入门见 [research-flow.md](../research-flow.md) 第三～五幕；数据侧（建档/清洗）依据见 [data-cleaning.md](data-cleaning.md)。
>
> **零词表绑定**（用户裁定）：本文的矩阵行 = 研究领域的模型/算法空间，不锚定引擎当前词表、不做一致性断言；引擎机制归叙事文档，能力签名以 MCP `tools/list` 为真源。
>
> **文献纪律**：同 [data-cleaning.md](data-cleaning.md)——每条主张与矩阵行带锚点（编号见文末[锚点一览](#锚点一览)，编号按本文档独立编排——跨文档同号不同文），逐条标注核验级别。

## 1. ML 模型矩阵（i.i.d. 表格数据）

| 模型族 | 归纳偏置 | 类别支持 | 缩放需求 | 数据量门槛 | 关键超参与正则 | 陷阱 | 文献 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 正则线性（ridge/lasso/逻辑回归） | 线性边界；lasso 稀疏化（L1 软阈值）[21] | 需编码 | **必须**（系数尺度敏感） | n≳10·特征数即可稳 | 正则强度 α（lambda）；lasso 在共线特征间随机择一 | 非线性关系下欠拟合；系数大小当特征重要性（尺度不齐时不可比） | [21][33] |
| kNN | 局部多数/平均；无训练、全靠距离 | 原生 | **必须**（距离被大量纲列支配） | 中大样本才稳（1-NN 渐近优良 ≠ 小样本优良） | k、距离度量、加权 | 高维诅咒（距离集中）；预测成本随 n 线性；ID 类特征 = 距离噪声 | [22] |
| 随机森林（RF） | bagging + 随机子空间决策树；方差削减 | 原生有序处理 | 不需要（阈值分裂对单调变换不变） | 小样本即可起跑（数百行已可用） | 树数（多即稳）、max_features（关键旋钮）、max_depth | 外推能力差（值域外输出钉在边界）；默认超参已强、调参收益薄——别把预算烧在 RF 上 | [23] |
| GBDT·XGBoost | 梯度提升 + 二阶导 + 正则化目标 [24] | 需编码 | 不需要 | 中小样本强（数百～百万） | learning_rate × n_estimators（联动）、subsample/colsample、min_child_weight、max_depth | lr 大 + 树少 = 欠拟合；早停轮数也是超参（须验证折内选） | [24][25] |
| GBDT·LightGBM | 直方图分裂 + GOSS/EFB 提速 [25] | 原生（`categorical_feature` 直喂，勿先行 one-hot 摊薄） | 不需要 | 中大样本（10⁴ 起、越大越占优） | num_leaves（比 depth 更贴切）、min_data_in_leaf（小样本防过拟合主旋钮） | 小样本（<~10³）上叶多易过拟合 | [25] |
| GBDT·CatBoost | 有序提升 + 目标统计编码内建（抗目标泄漏设计）[26] | **原生**（ordered target statistics） | 不需要 | 中小样本强；小样本上类别列优势最明显 | depth（浅）、l2_leaf_reg、one_hot_max_size | 类别列喂原生接口而非自编码；迭代序敏感性（有序提升吃随机性） | [26] |

数据量门槛速查（i.i.d. 表格、常规特征数）：**几十～几百行**正则线性/浅 GBDT/CatBoost（类别列多时）；**几百～几千行**树族主场（Gorishniy 等多基准下 DL 与 GBDT 互有胜负、无一制霸 [27]）；**~10⁴ 行级**树族对多数 DL 架构仍有系统优势（Grinsztajn 等基准结论 [28]）；更大规模进入 DL 可竞争区（特征工程 + 架构得当，见 §2.1）。

## 2. DL 模型矩阵

| 架构 | 归纳偏置 | 适合数据形态 | 数据量门槛 | 可解释面 | 陷阱 | 文献 |
| --- | --- | --- | --- | --- | --- | --- |
| MLP（FNN） | 通用逼近（宽度/深度换表达）[29] | 任意数值化输入 | 表格上需强正则才与树族抗衡 | 弱（权重归因不直观） | 表格上裸 MLP 常输 GBDT——不是容量不够而是正则没调对 [30] | [29][30] |
| 正则化 MLP | 同 MLP + 13 种正则鸡尾酒联合搜索 [30] | 同 MLP | 数千行起即可竞争 | 同 MLP | 正则组合本身是大型搜索空间——搜索预算要留给它 | [30] |
| ResNet 型（残差 MLP） | 残差连接缓和深网优化 | 同 MLP | 中等（NeurIPS'21 基准中作为强基线复兴）[27] | 弱 | 常被跳过不试——它是「最便宜的 DL 强基线」 | [27] |
| FT-Transformer | 特征 token 化 + attention 建模特征交互 [27] | 中大样本表格 | ~10⁴ 行级（小样本上注意力欠约束） | 注意力权重可看但归因有争议 | 小样本不稳；数值特征需配嵌入（见数值嵌入行） | [27][31] |
| TabNet | 序贯注意力选特征（稀疏化决策步）[32] | 特征选择叙事强的表格 | 中大样本 | 内建特征归因面具 | 官方对比的调参协议曾受质疑——自行复基准再采信；小样本优势不明显 | [32] |
| NODE | 神经遗忘决策集成（可微树集成）[34] | 表格 | 中等 | 树结构可读 | 训练敏感（初始化/学习率）；收益不稳 | [34] |
| TabPFN（先验数据拟合网络） | 合成先验上预训练的上下文学习分类器 [35] | **小样本分类**（176 数据集实证：训练集 ≤~3000 行仍平均领先 [36]；类别/特征规模上限随版本放宽 [37]） | 无需训练（秒级推断） | 黑盒（预训练先验） | 域外特征类型（类别需序编码、文本/缺失需预处理）表现退化 | [35][36][37] |
| 数值嵌入（piecewise-linear / periodic） | 标量 → 高维嵌入再入主干 [31] | 与任意表格主干可组合（MLP 受益最明显） | 中等起 | 弱 | 嵌入可大幅改变 ML/DL 相对表现——比较实验必须声明嵌入用法 | [31] |
| 1D-CNN | 平移不变（局部模式滑动共享权重） | 定长信号/光谱/序列片段（如光谱、振动、时序窗口） | 中等（卷积先验省样本） | 卷积核可视化（局部模式） | i.i.d. 表格（列无序）不适用——列序是它的前提不是它的发现 | [38] |
| RNN（LSTM/GRU） | 门控记忆的状态序列建模 [39][40] | 变长序列（有序依赖真正的时序） | 中大样本 | 隐状态可检查但弱 | 表格 i.i.d. 任务硬排序列 = 引入假时序；长程依赖仍衰减 | [39][40] |

架构选型先问**数据形态**：i.i.d. 表格 → §1 树族优先；定长信号 → 1D-CNN；真序列 → RNN/Transformer；表格但量大特征交互复杂 → FT-Transformer/数值嵌入线。

### 2.1 何时值得上 DL（实证证据）

- **~10⁴ 行级以下的中等表格，树族仍是系统性强者**：Grinsztajn 等（45 数据集统一调参协议）结论——树族在中等规模表格上保持领先，差距归因于归纳偏置（不规则目标函数、无信息特征鲁棒性、数据保持方向性）[28]；
- **「NN vs GBDT」之争被高估**：McElfresh 等（176 数据集 × 19 算法）发现大量数据集上两者差距可忽略，或「对 GBDT 做轻量调参」比「选哪个阵营」更影响结果；GBDT 对偏态/重尾特征分布等不规则性显著更稳 [36]；
- **小样本例外是 TabPFN**：训练集 ≤3000 行时平均超过全部对手（含强调参 GBDT）[35][36]；
- **DL 的翻盘点**：表征学习（数值嵌入 [31]、预训练 [37]）与强正则搜索 [30]——不是「换个架构」而是「换表征 + 换训练法」；综述见 Borisov 等 [41]。
- 总纲提醒（律二·基线参照）：任何 DL 尝试都必须与同场树族基线对照后才有结论——无免费午餐定理保证不存在全域占优的学习器 [42]。

## 3. CV 与超参搜索（CV-HPO）

### 3.1 嵌套 CV 与选择偏差

**模型选择与性能估计必须两层分开**：用同一套 CV 既选超参又报性能 = 选择偏差——搜索得越狠，乐观偏差越大（在纯噪声数据上也能「搜出」低于期望的误差）[43][44]。纪律：

- 内层选（θ*），外层估（u）——嵌套 CV 的外层估计近似无偏 [44]；
- holdout 只读一次的纪律是同一件事的终局形态：终选之后的唯一一次测试 = 外层估计的极限特例（见 [research-flow.md](../research-flow.md) 律一）；
- 全流程入折：预处理、填充、特征选择、缩放——凡是「拟合型」变换都在内层折内完成（Kapoor & Narayanan 的泄漏分类学把「预处理先于划分」列为一级泄漏 [12]）。

### 3.2 折间方差与配对比较的正确姿势

- **K 折方差没有无偏估计** [45]：折间共享训练集，协方差结构使经验方差系统性低估——不要把「折间 std 当标准误」去造置信区间；
- **配对比较用修正 t**：Nadeau–Bengio 修正（分母含 k+1/(k−1) 项补偿折间相关）[46]；经典 Dietterich 实验早已表明「多重随机划分的配对 t」与「比例差检验」第一类错误失控 [47]；
- **K 折的选择**：大样本 5 折够（方差/成本折中，教材通行建议 [33]）；小样本重复 K 折降方差但注意重复不改系统性偏差 [44]；
- 单一标准差规则（one-standard-error）：在最优分数一个标准差内的最简模型——简约优先的成文做法 [33]。

### 3.3 搜索算法形态对比

| 搜索形态 | 适用 | 预算纪律 | 文献 |
| --- | --- | --- | --- |
| 网格搜索 | 维度 ≤2 且每维离散少数档 | 网格成本 = 各维笛卡尔积——维度稍高即爆；**低效有证**：无关维度稀释预算（随机搜索同预算更优，因重要维度常只少数几个） | [48] |
| 随机搜索 | 默认起步；混合空间（连续×离散）自然支持 | 同预算下命中好配置概率 ≥ 网格；预算花在重要维度的密度上 | [48] |
| TPE（树结构 Parzen 估计） | 中等预算（10²～10³ trial）；Optuna 默认 [50] | 顺序模型无关优化（SMBO）族——好区域加密采样；对条件空间（级联超参）原生支持 | [49][50] |
| 高斯过程 BO | 目标评估昂贵（每次 trial 训练很久）+ 维度低中 | 样本效率最高但扩展性弱；采集函数本身有超参 | [51] |
| Hyperband/早停族 | 训练可中途评估（epoch/ boosting 轮数做资源轴）；大并行、预算不定 | 早期淘汰劣配置（纯探索老虎机理论）；对「预算未知先跑起来」的场景最贴 | [52] |
| BOHB | BO + Hyperband 混合 | 大预算下兼顾早期淘汰与稠密区域建模 | [53] |

选型一句话：**预算 < 数十 trial 起步随机 [48]；标准 Optuna 工作流走 TPE [49][50]；trial 很贵走 GP-BO [51]；可早停 + 大并行走 Hyperband/BOHB [52][53]**。搜索不是越狠越好——同一验证面上搜得越狠，选择偏差越大（§3.1），搜索预算与嵌套结构必须配套声明。

## 4. 特征工程

### 4.1 编码矩阵（类别 → 数值）

| 编码 | 语义 | 适用 | 泄漏风险 | 陷阱 | 文献 |
| --- | --- | --- | --- | --- | --- |
| one-hot | 每类一维指示 | 低基数（≲ 10–100） | 低（类别集合可全量定） | 高基数下维度爆炸 + 树族分裂被摊薄（树族宁愿原生类别） | [33] |
| 序数编码 | 类别 → 有序整数 | **真有序**类别（极性序、等级） | 低 | **假序注入**：无序类别硬编码 = 给模型编造距离；序须领域确认 | [33] |
| 目标编码 | 类别 → 折内目标统计（平滑经验贝叶斯）[54] | 高基数（邮编/型号/ID 类信号列） | **高**：目标统计必须折内算（全量算 = 目标泄漏重灾区）；CatBoost 将这类目标统计的预测偏移作为核心问题分析并给出有序统计防御 [26] | 罕见类的统计噪声（须平滑/最小样本门）；目标编码泄漏是竞赛/刷分场景最常见的幻觉增益来源之一 | [54][26] |
| 频数编码 | 类别 → 出现次数 | 中高基数 | 低（频数可全量算，但严格折内更稳） | 频数与目标无关时是噪声特征 | [33] |
| 哈希编码 | 类别 → 固定维哈希 | 超高基数/在线流式 | 低 | 碰撞不可解释；哈希空间大小是新超参 | [33] |

### 4.2 缩放与变换矩阵（数值列）

| 变换 | 语义 | 何时做 | 文献 |
| --- | --- | --- | --- |
| standard（z 标准化） | 均值 0 方差 1 | 线性族/NN/距离类**必做**；本身对离群敏感（见 data-cleaning.md §2.3） | [33] |
| robust（中位数/IQR 标定） | 抗离差的标准化 | 有离群残影的列配线性/NN | [33] |
| min-max | 压到 [0,1] | 需要有界输出时（NN 某些层） | [33] |
| log1p/log | 压右偏/乘性尺度 | 计数、金额、强度类长尾列——**先问零/负值** | [33] |
| Box-Cox/Yeo-Johnson | 参数化正态化族 | 需要正态性假设的下游（线性推断）；λ 由数据估 → **折内拟合** | [55] |

### 4.3 变换的模型相关性（做过无意义的、别漏做关键的）

- **树族对单调变换不变**（阈值分裂只看序）[23][24]：对 RF/GBDT 做 log/standard 是**无关键**——不伤害但也无收益（分裂点等价），除非变换改变 ties/缺失语义；
- **缩放对树族无关、对线性/NN/距离类关键** [21][33]：同一 pipeline 对不同模型族的最小必要变换集不同——盲目全量标准化是浪费，漏做才是事故；
- **非线性特征（交互项、比率）对树族是先验**：GBDT 能学交互但深树才学得到——显式构造领域交互（如 浓度×温度）在样本小时常比让树自己学便宜。

### 4.4 pipeline 化防泄漏纪律

一切「拟合型」变换（缩放的 μ/σ、目标编码的统计、填充值、Box-Cox 的 λ、特征选择的入选集）**只能在训练折内拟合、折外只 transform** [12][43]。成文三条：

1. 预处理先于划分 = 一级泄漏（泄漏分类学 [12]），管线对象（fit on train / transform on val）是结构性防御；
2. 特征选择的入选集是「学出来的统计量」——折外数据参与选特征同样是泄漏 [43]；
3. 验收面：任何「不折叠的预处理让分数大涨」都可疑——涨幅的合理默认解释是泄漏而非信息（同 [research-flow.md](../research-flow.md) 律一）。

## 5. 模型评估

### 5.1 任务 ↔ 指标

| 任务形态 | 推荐主指标 | 理由与陷阱 | 文献 |
| --- | --- | --- | --- |
| 不平衡二分类 | PR 曲线 / PR-AUC | ROC-AUC 在类先验悬殊时虚高（假阳性率分母被负类淹没）；PR 面对不平衡更诚实 | [56] |
| 任意二分类报告 | MCC | 观察与预测二值化的相关系数——混淆矩阵四格全纳入、类间尺寸均衡（accuracy/F1 各有盲区，F1 忽略真负类）；单值可比 | [57] |
| 回归 | R²/MAE/RMSE 按业务口径 | 报告**原始单位**的 MAE（组会可读）；R² 对照外部基线才有意义 | [33] |
| 概率质量任务 | 校准曲线/ECE | 准确的分类器可能是差的概率估计器（现代网络系统性过自信）| [58] |

指标先于训练定（建模协议期冻结）——事后换指标 = 对测试集二次拟合（选择偏差的指标形态 [43]）。

### 5.2 holdout 一次性与泄漏分类学

- **holdout 是一次性资源**：终选后恰一次触碰；反复对 holdout 求梯度 = 适应性过拟合，基准数据上被证明几十轮内即失效 [59]；
- **泄漏分类学** [12]：Kapoor & Narayanan 按发生环节给出系统分类（L1 数据准备、L2 特征工程、L3 含时间在内的划分泄漏等）。自查三问：①全流程（预处理/填充/编码/选特征）是否都折内完成（L1/L2）？②test 集是否只被终评碰过一次——反复触碰的代价见 [59]？③同一验证面是否被反复用来做选择（§3.1）？共同纪律：**数据流图先画出来，凡箭头从 test/折外指向拟合面的都是泄漏**；
- 时序数据：随机 K 折打破时间序 = 时间型划分泄漏 [12]；按时间切分（或分组切分）是硬要求。

### 5.3 无显著差异判读（差距小于折间波动时）

- **描述性规则**（默认）：差距 < 折间波动 → 判「无显著差异」，不强行排名（[research-flow.md](../research-flow.md) 律三）——理由的统计学根据是折间方差无无偏估计、朴素 t 失控 [45][46][47]；
- **要下统计结论时用修正配对 t**（Nadeau–Bengio）[46]，前提是同折同种子配对；
- **要主张「等价」时用等价检验（TOST）**：设业务上的最小重要差（等价界），双侧单边检验同时拒绝两端 [60][61]——「检验不显著」≠「无差异」（不显著只是没证据，等价才是正面主张）；
- 边界纪律：描述性规则是**保守的工程默认**，等价检验是**更强的统计主张**——报告里写清用的是哪一个。

### 5.4 对照系与泰勒图

- **对照系三层**：领域基线（经验公式/文献数）、平凡基线（均值预测/多数类）、同场多模型——缺层的「SOTA」是孤立数字（律二）[42]；
- **泰勒图**一图三量：相关系数（形状）、标准差比（幅度）、RMSE（总误差）的三角关系 [62]——多模型对比时比单指标排序信息量大：两个 RMSE 相近的模型可以误差结构完全不同（一个偏幅度一个偏相位）。

### 5.5 解释面判读（SHAP 读法与陷阱）

- **读法**：特征归因是**条件性陈述**（该观测下该特征的贡献），不是因果效应 [63]；汇总图看分布（哪些特征稳定重要）、依赖图看形状（单调/峰形/U 形）、单样本力图讲个案；
- **陷阱**：①相关特征间归因可互相转移（SHAP 的一致性保证的是加性一致不是特征语义稳定）；②依赖图上的「形状」可能是耦合特征的假象（交互项检查：着色第二特征）；③归因方向与业务直觉冲突是**最高价值信号**——它要么抓到了真发现、要么抓到了泄漏/缺陷，两者都值得回流（[research-flow.md](../research-flow.md) 第五幕）；
- **校准面**：概率模型的解释先看校准曲线（可靠性图）——过自信模型 SHAP 再漂亮也不能当概率用 [58]；
- **误差切片**：聚合指标会掩盖局部塌陷（某切片上模型系统性差）；行为测试（CheckList 式：按能力×测试类型的矩阵构造切片用例）把「误差集中在哪」变成系统性检查而非运气 [64]。

### 5.6 预测区间（conformal 研究线）

点预测之外报告区间时，分布无关的保形预测（split conformal）以「交换性」为唯一假设给出有限样本覆盖保证 [65][66]：非一致性分数在校准集上的分位数即区间宽度——**校准集同样不可被训练过程碰过**（否则覆盖保证失效；与 §3.1 同一纪律）。回归区间（±带宽）与分类的预测集（top-k 集合）同框架。

### 5.7 可复现清单

终局交付前的最小检查面（对照 ML 生产就绪清单 [67] 裁剪为研究场景）：

- [ ] 种子固定 + 全部随机源声明（数据划分/初始化/搜索采样）；
- [ ] 血缘可重放：原始数据 → 清洗计划 → 特征管线 → 训练配置全程留痕，第三人可端到端重跑；
- [ ] 划分纪律成文：划分方式（随机/分组/时序）、holdout 触碰次数（=1）；
- [ ] 指标口径 + 外部基线 + 折间波动（或修正 t 结果）三件套齐；
- [ ] 泄漏自检：数据流图过一遍 §5.2 自查三问；
- [ ] 解释面结论标注「关联非因果」；校准状态已检查（若用概率）。

## 锚点一览

| # | 题录 | 核验级别 |
| --- | --- | --- |
| [12] | Kapoor, S. & Narayanan, A. *Leakage and the reproducibility crisis in machine-learning-based science*. Patterns 4(9):100804, 2023 | 摘要级（Crossref/DOI 10.1016/j.patter.2023.100804；与 data-cleaning.md [12] 同条） |
| [21] | Tibshirani, R. *Regression Shrinkage and Selection via the Lasso*. JRSS-B 58(1):267–288, 1996 | 摘要级（Crossref/DOI 10.1111/j.2517-6161.1996.tb02080.x） |
| [22] | Cover, T. & Hart, P. *Nearest Neighbor Pattern Classification*. IEEE Trans. Inf. Theory 13(1):21–27, 1967 | 摘要级（Crossref/DOI 10.1109/TIT.1967.1053964） |
| [23] | Breiman, L. *Random Forests*. Machine Learning 45(1):5–32, 2001 | 摘要级（Crossref/DOI 10.1023/A:1010933404324） |
| [24] | Chen, T. & Guestrin, C. *XGBoost: A Scalable Tree Boosting System*. KDD 2016:785–794 | 摘要级（Crossref/DOI 10.1145/2939672.2939785）；GBM 理论源流：Friedman, J. H. *Greedy Function Approximation: A Gradient Boosting Machine*. Ann. Statist. 29(5):1189–1232, 2001（摘要级，DOI 10.1214/aos/1013203451） |
| [25] | Ke, G. et al. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NIPS 30:3146–3154, 2017 | 摘要级（NeurIPS proceedings 多源交叉） |
| [26] | Prokhorenkova, L. et al. *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS 2018（arXiv:1706.09516） | 摘要级（arXiv abs 页） |
| [27] | Gorishniy, Y., Rubachev, I., Khrulkov, V. & Babenko, A. *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021（arXiv:2106.11959） | 摘要级（arXiv abs 页） |
| [28] | Grinsztajn, L., Oyallon, E. & Varoquaux, G. *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 2022 Datasets & Benchmarks（arXiv:2207.08815） | 摘要级（arXiv abs 页） |
| [29] | Hornik, K., Stinchcombe, M. & White, H. *Multilayer Feedforward Networks are Universal Approximators*. Neural Networks 2(5):359–366, 1989 | 摘要级（Crossref/DOI 10.1016/0893-6080(89)90020-8） |
| [30] | Kadra, A., Lindauer, M., Hutter, F. & Grabocka, J. *Well-tuned Simple Nets Excel on Tabular Datasets*（arXiv:2106.11189, 2021） | 摘要级（arXiv abs 页） |
| [31] | Gorishniy, Y., Rubachev, I. & Babenko, A. *On Embeddings for Numerical Features in Tabular Deep Learning*. NeurIPS 2022（arXiv:2203.05556） | 摘要级（arXiv abs 页） |
| [32] | Arik, S. Ö. & Pfister, T. *TabNet: Attentive Interpretable Tabular Learning*. AAAI 2021（arXiv:1908.07442） | 摘要级（arXiv abs 页） |
| [33] | Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*（2nd ed., Springer, 2009） | 书目级（Crossref/DOI 10.1007/978-0-387-84858-7） |
| [34] | Popov, S., Morozov, S. & Babenko, A. *Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data*. ICLR 2020（arXiv:1909.06312） | 摘要级（arXiv abs 页） |
| [35] | Hollmann, N., Müller, S., Eggensperger, K. & Hutter, F. *TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second*. ICLR 2023（arXiv:2207.01848；OpenReview forum cp5PvcI6w8_） | 摘要级（arXiv abs 页题名/作者 + OpenReview ICLR 2023 收录多源交叉；注意 PMLR v202 无此文——检索摘要曾误作 ICML 2023，一手页已证伪） |
| [36] | McElfresh, D. et al. *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* NeurIPS 2023（arXiv:2305.02997） | 摘要级（arXiv abs 页——TabPFN ≤3000 行训练集约束出自其摘要） |
| [37] | Hollmann, N. et al. *Accurate predictions on small data with a tabular foundation model*. Nature 637:319–326, 2025 | 摘要级（Nature 页多源交叉） |
| [38] | Kiranyaz, S. et al. *1D Convolutional Neural Networks and Applications: A Survey*. Mech. Syst. Signal Process. 151:107398, 2021 | 摘要级（Crossref/DOI 10.1016/j.ymssp.2020.107398） |
| [39] | Hochreiter, S. & Schmidhuber, J. *Long Short-Term Memory*. Neural Computation 9(8):1735–1780, 1997 | 摘要级（Crossref/DOI 10.1162/neco.1997.9.8.1735） |
| [40] | Cho, K. et al. *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP 2014:1724–1734（GRU） | 摘要级（Crossref/DOI 10.3115/v1/D14-1179） |
| [41] | Borisov, V. et al. *Deep Neural Networks and Tabular Data: A Survey*. IEEE TNNLS 35(6):7499–7519, 2024 | 摘要级（Crossref/DOI 10.1109/TNNLS.2022.3229161） |
| [42] | Wolpert, D. H. & Macready, W. G. *No Free Lunch Theorems for Optimization*. IEEE Trans. Evol. Comput. 1(1):67–82, 1997 | 摘要级（Crossref/DOI 10.1109/4235.585893） |
| [43] | Cawley, G. C. & Talbot, N. L. C. *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR 11:2079–2107, 2010 | 摘要级（JMLR 一手页） |
| [44] | Varma, S. & Simon, R. *Bias in Error Estimation when Using Cross-Validation for Model Selection*. BMC Bioinformatics 7:91, 2006 | 摘要级（Crossref/DOI 10.1186/1471-2105-7-91） |
| [45] | Bengio, Y. & Grandvalet, Y. *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*. JMLR 5:1089–1105, 2004 | 摘要级（JMLR 一手页） |
| [46] | Nadeau, C. & Bengio, Y. *Inference for the Generalization Error*. Machine Learning 52(3):239–281, 2003 | 摘要级（Crossref/DOI 10.1023/A:1024068626366） |
| [47] | Dietterich, T. G. *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. Neural Computation 10(7):1895–1923, 1998 | 摘要级（Crossref/DOI 10.1162/089976698300017197） |
| [48] | Bergstra, J. & Bengio, Y. *Random Search for Hyper-Parameter Optimization*. JMLR 13:281–305, 2012 | 摘要级（JMLR 一手页） |
| [49] | Bergstra, J., Bardenet, R., Bengio, Y. & Kégl, B. *Algorithms for Hyper-Parameter Optimization*. NIPS 24:2546–2554, 2011 | 摘要级（NeurIPS proceedings 多源交叉） |
| [50] | Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. *Optuna: A Next-generation Hyperparameter Optimization Framework*. KDD 2019:2623–2631 | 摘要级（Crossref/DOI 10.1145/3292500.3330701） |
| [51] | Snoek, J., Larochelle, H. & Adams, R. P. *Practical Bayesian Optimization of Machine Learning Algorithms*. NIPS 25:2951–2959, 2012 | 摘要级（NeurIPS proceedings 多源交叉） |
| [52] | Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. & Talwalkar, A. *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization*. ICLR 2017（扩展版 JMLR 18(1), 2018） | 摘要级（JMLR/OpenReview 多源交叉） |
| [53] | Falkner, S., Klein, A. & Hutter, F. *BOHB: Robust and Efficient Hyperparameter Optimization at Scale*. ICML 2018, PMLR 80 | 摘要级（PMLR v80 一手页） |
| [54] | Micci-Barreca, D. *A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems*. SIGKDD Explorations 3(1):27–32, 2001 | 摘要级（Crossref/DOI 10.1145/507533.507538） |
| [55] | Box, G. E. P. & Cox, D. R. *An Analysis of Transformations*. JRSS-B 26(2):211–243, 1964 | 摘要级（Crossref/DOI 10.1111/j.2517-6161.1964.tb00553.x） |
| [56] | Saito, T. & Rehmsmeier, M. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE 10(3):e0118432, 2015 | 摘要级（Crossref/DOI 10.1371/journal.pone.0118432） |
| [57] | Chicco, D. & Jurman, G. *The Advantages of the Matthews Correlation Coefficient (MCC) over F1 Score and Accuracy in Binary Classification Evaluation*. BMC Genomics 21:6, 2020 | 摘要级（Crossref/DOI 10.1186/s12864-019-6413-7） |
| [58] | Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. *On Calibration of Modern Neural Networks*. ICML 2017（arXiv:1706.04599） | 摘要级（arXiv abs 页 comments "ICML 2017"） |
| [59] | Blum, A. & Hardt, M. *The Ladder: A Reliable Leaderboard for Machine Learning Competitions*. ICML 2015, PMLR 37:1006–1014（arXiv:1502.04585） | 摘要级（PMLR v37 一手页 + Semantic Scholar venue 交叉；初稿误作 NeurIPS 2015——一手页更正） |
| [60] | Schuirmann, D. J. *A Comparison of the Two One-Sided Tests Procedure and the Power Approach for Assessing the Equivalence of Average Bioavailability*. J. Pharmacokinetics and Biopharmaceutics 15(6):657–680, 1987 | 摘要级（Crossref/DOI 10.1007/BF01068419） |
| [61] | Lakens, D. *Equivalence Tests: A Practical Primer for t-Tests, Correlations, and Meta-Analyses*. Soc. Psychol. Personal. Sci. 8(4):355–362, 2017 | 摘要级（PsyArXiv 预印本页 + 期刊卷期多源交叉） |
| [62] | Taylor, K. E. *Summarizing Multiple Aspects of Model Performance in a Single Diagram*. J. Geophys. Res. Atmos. 106(D7):7183–7192, 2001 | 摘要级（Crossref/DOI 10.1029/2000JD900719） |
| [63] | Lundberg, S. M. & Lee, S.-I. *A Unified Approach to Interpreting Model Predictions*. NeurIPS 2017（arXiv:1705.07874） | 摘要级（arXiv abs 页 comments "To appear in NIPS 2017"） |
| [64] | Ribeiro, M. T., Wu, T., Guestrin, C. & Singh, S. *Beyond Accuracy: Behavioral Testing of NLP Models with CheckList*. ACL 2020:4902–4912 | 摘要级（Crossref/DOI 10.18653/v1/2020.acl-main.442） |
| [65] | Vovk, V., Gammerman, A. & Shafer, G. *Algorithmic Learning in a Random World*（2nd ed., Springer, 2022） | 书目级（Crossref/DOI 10.1007/978-3-031-06649-8） |
| [66] | Angelopoulos, A. N. & Bates, S. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511, 2022（Foundations and Trends in ML, 2023） | 摘要级（arXiv abs 页） |
| [67] | Breck, E., Cai, S., Nielsen, E., Salib, M. & Sculley, D. *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction*. IEEE BigData 2017:1123–1132 | 摘要级（IEEE 多源交叉） |
