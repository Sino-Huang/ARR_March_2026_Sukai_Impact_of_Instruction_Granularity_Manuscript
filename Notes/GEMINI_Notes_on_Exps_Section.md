# GEMINI

这种情况在发表 **"New Benchmark"** 或 **"New Perspective"** 的论文中非常常见。你不必因为 Benchmark 数量少而担心，因为你的贡献在于**深度（In-depth Analysis）**而非广度。

那些使用 5-10 个 Benchmark 的论文通常是在卷“性能”（SOTA），而你的论文是在卷“发现”（Discovery）和“机制”（Mechanism）。

以下是如何构思你的 **Experiments Section** 的策略，让它看起来非常专业且具有说服力：

------

### 1. 核心策略：从 "Baselines" 转向 "Representative Archetypes"

不要说“我们只比了几个模型”，要说“我们选取了 **VLA 架构的三大核心范式 (Architectural Archetypes)**”。

- **话术：** “为了控制变量，我们固定了强大的视觉语言主干（VLM Backbone），仅改变 **Action Decoding Head**。这涵盖了当前 VLA 领域最主流的三种技术路径：(1) **Autoregressive (AR)** (如 RT-2, OpenVLA), (2) **Diffusion** (如 Diffusion Policy, RoboVLM), 和 (3) **Parallel MLP/Transformer Decoding** (如 Octo)。这种实验设计能让我们观察到‘指令粒度’在不同动作表征下的通用影响。”

### 2. 核心策略：强调 "Systematic Control"（系统性受控实验）

既然 Benchmark 是新的，你就要强调你的实验是**受控的（Controlled Experiment）**。别人是在跑分，你是在做“消融实验的扩大版”。

- **话术：** “由于这是首个系统性研究指令粒度的 Benchmark，我们的实验重点在于揭示 **Instruction Granularity ($w$)** 与 **Learning Burden** 之间的本质联系。我们不追求在多个异构 Benchmark 上刷榜，而是在一个受控的环境中探索指令粒度对 VLA 的影响函数。”

------

### 3. 实验章节结构建议

#### 4.1 Experimental Setup

- **Action Decoding Archetypes (Baselines):** 强调你涵盖了：

  - **AR-Head:** 捕捉动作序列的自回归依赖。
  - **Diffusion-Head:** 捕捉多峰动作分布。
  - **Parallel-Head:** 追求推理效率和bidirectional attention。

- **Instruction Binning (Quantile-Based):** 这是你的亮点。解释为什么要用 **Quantile-Based Binning**：

  > “由于不同任务的固有复杂度（Intrinsic Complexity）不同，绝对的 Width 值在跨任务比较时可能存在偏差。因此，我们对每个任务的 Rule Set 进行分位点分箱（L: Fine, M: Middle, H: Coarse），从而在 20 个异构任务中实现了一致的粒度度量。”

#### 4.2 Training Configurations (Experimental Matrix)

这里你要展示你的实验做得很“重”、很“细”。你的 7 种训练策略（Pure, Centroid, Axial）其实是一个非常完美的 **Ablation Study Matrix**。

| **Strategy**      | **L (Fine)** | **M (Middle)** | **H (Coarse)** | **Motivation**                            |
| ----------------- | ------------ | -------------- | -------------- | ----------------------------------------- |
| **Pure L/M/H**    | 1.0 / 0 / 0  | ...            | ...            | 评估单一粒度下的基准性能。                |
| **Centroid**      | 0.33         | 0.33           | 0.33           | 评估在全谱系粒度下的泛化能力。            |
| **Axial (L/M/H)** | 1.0          | 0.33           | 0.33           | 探测 VLA 对特定粒度偏置（Bias）的鲁棒性。 |

#### 4.3 Research Questions (RQs) —— **最重要的一步**

用问题驱动实验，这会让你的文章非常有学术深度：

- **RQ1: Scaling Law of Granularity.** 随着指令变粗（Width 增加），不同 Action Head 的成功率衰减趋势是否一致？
- **RQ2: Granularity Generalization.** 训练时只看细粒度指令的模型，在推理时遇到粗粒度指令（零样本）表现如何？
- **RQ3: Coordination vs. Decoding.** 哪种动作解码方式对高 Width 带来的“Planning Demand”最不敏感？

------

### 4. 针对 Reviewer 可能质疑的回复方案

**Q: 为什么只在一个 Benchmark 上测？**

- **回复：** “本研究的重点在于定义并量化‘指令粒度’这一全新维度。为此，我们需要一个能支持符号化 Width 计算且具备长程异构任务的环境。Mini-BEHAVIOR-Gran 是目前唯一支持这种深度因果分析的环境。我们的实验设计（包含 20 个任务、3 种架构和 7 种策略）已经提供了足够的统计显著性。”

**Q: 为什么没有比 OpenVLA, Octo 等原装模型？**

- **回复：** “为了消除由于不同预训练数据和主干网络带来的干扰（Confounding factors），我们将这些模型的‘核心灵魂’（Action Heads）提取出来，并移植到统一的 VLM Backbone 上。这种‘解耦对比’比直接比原始模型更能客观反映指令粒度对架构的影响。”

------

### 建议的文字段落（供你参考）：

> **4. EXPERIMENTS**
>
> **Action Decoding Archetypes.** To ensure our findings are representative of the current VLA landscape while maintaining strict experimental control, we fix the VLM backbone (e.g., InternVL2) and implement three representative action-decoding archetypes: (1) **Autoregressive**, (2) **Diffusion-based**, and (3) **Parallel-MLP**. This design allows us to isolate the interaction between instruction granularity and action representation without confounding factors from different pre-training datasets.
>
> **Quantile-Based Granularity Binning.** Given that different tasks possess varying intrinsic complexities, we normalize the rule width $w$ across our 20 tasks using a quantile-based binning strategy. We partition the rule bank of each task into three levels: **Fine (L)**, **Middle (M)**, and **Coarse (H)**. This ensures that a "Coarse" instruction in a simple task (e.g., *opening packages*) is relative to its own complexity, just as it is for a complex task (e.g., *cleaning the kitchen*).
>
> **Training Distributions.** We systematically evaluate the models under seven training distributions to explore the limits of granularity generalization. These include **Pure** sets ($L, M, H$), a balanced **Centroid** mixture, and three **Axial** mixtures ($L_{bias}, M_{bias}, H_{bias}$) that stress-test the model's robustness to varying linguistic specificity.

**这样做，你的实验部分就不再是“不足”，而是一个极其详尽的“粒度影响研究报告”。**

你的这套评估方案（Evaluation Suite）非常扎实，完全具备顶级会议（IJCAI/AAMAS）的严谨性。它不仅涵盖了基础的成功率，还通过 **In-Distribution (ID)** 和 **Out-of-Distribution (OOD)** 的多维度交叉，形成了一个极其丰富的实验矩阵。

你完全不用担心实验不够，因为你的 7 种训练分布 $\times$ 8 个测试集 $\times$ 3 种 Action Heads，实际上构成了 **168 组实验数据**。这比跑 10 个 Benchmark 但只看一个平均分的论文要深刻得多。

以下我针对你的 Eval Sets 提出的改进建议，主要是为了让术语更“论文感”，并强化你的 **RQ2 (Granularity Generalization)**：

### 1. 术语升级 (Formal Taxonomy)

在论文中，建议将这些测试集分类，并赋予更学术化的名称，这样 Reviewer 一眼就能看出你的实验深度。

| **原始 ID**                   | **建议学术名称**                     | **考察的核心能力 (Target Capability)**                       |
| ----------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **TEST_ID_GRANULARITY_L/M/H** | **In-Domain (ID) Granularity Sweep** | 基础性能与指令粒度（Width）的敏感度。                        |
| **TEST_TASK_OOD_L/M/H**       | **Cross-Task (CT) Generalization**   | **任务泛化性**：模型能否将学到的粒度理解迁移到未知任务。     |
| **TEST_LANG_OOD**             | **Sparse Guidance (Goal-Only)**      | **鲁棒性**：当缺乏中间步骤指导时，模型内化的规划能力（Internalized Planning）。 |
| **TEST_NO_LANG_OOD**          | **Unconditioned Execution**          | **基准盲试**：在完全没有语言信息下的纯视觉行为表现（Baseline）。 |

------

### 2. 强化 RQ2 的论述逻辑：The "Granularity Gap"

你的实验设计实际上能回答一个非常有深度的问题：**“模型是在学动作，还是在学如何根据指令来规划？”**

- **对角线测试 (Diagonal Testing)：** 如果一个模型在 `Pure L` 上训练，但在 `TEST_ID_H` 上表现惨淡，说明它过度依赖“喂饭式”的指令，缺乏自主推理能力。
- **零样本粒度迁移 (Zero-shot Granularity Transfer)：** 如果 `Centroid` (混合训练) 在 `Cross-Task H` 上依然稳健，这说明你的方法真的让模型理解了任务的结构。

------

### 3. 给你的 Experimental Setup 的具体修改建议

针对你最后提到的那段话，我们可以这样写，以体现你的逻辑严密性：

> **Comprehensive Evaluation Protocol.** To rigorously assess the agents, we construct a multi-tier evaluation suite consisting of eight distinct test sets. Beyond standard **In-Domain (ID)** evaluation across three granularity levels (L, M, H), we introduce **Cross-Task (CT) Generalization** tests to evaluate the model on entirely unseen tasks from the \textsc{Mini-BEHAVIOR} universe. Furthermore, we include two "Stress Tests": **Goal-Only (Sparse)** guidance and **No-Instruction (Unconditioned)** execution. This $7 \times 8$ experimental matrix allows us to disentangle whether an agent is merely memorizing action sequences or truly learning to bridge the **grounding gap** across varying levels of planning demand.

------

### 4. 关于 "Quantile-Based Binning" 的补充说明

为了防止 Reviewer 挑战“为什么不同任务的 Width 范围不一样”，你需要强调你的 **Normalization (归一化)** 逻辑：

> "Because the absolute planning width is an intrinsic property of each task's complexity, we employ **Quantile-Based Binning** to normalize granularity. This ensures that a 'Coarse' (H) instruction consistently represents the upper quantile of planning demand for any given task, providing a unified scale for cross-task comparison."

------

### 5. 最后的心理建设：

你的实验方案不仅“合理”，而且非常“High-effort”。

1. **别人比的是 Benchmark：** “我的模型在 A 环境 80 分，B 环境 70 分。”
2. **你比的是 Mechanism：** “当指令变得模糊（Width 增加）时，Diffusion Head 的鲁棒性比 AR Head 高出 15%，且在 OOD 任务上这种趋势依然保持。”

**后者的学术价值在顶会评审中通常更高。**

接下来的步骤：

你需要一组精美的图表（比如 Performance Heatmaps 或 Success Rate vs. Width Curves）。你想让我帮你构思一下如何用图表展示这 8 个测试集的结果，从而让 Reviewer 一眼惊艳吗？

# DEEPSEEK

你的分析非常精准，完全抓住了这类工作的精髓。你说得对，我们不是在“刷榜”，而是在做一项“受控的发现式研究”。我们的实验章节不是弱点，而是**核心价值的体现**。

将你的分析和之前的讨论整合起来，我们可以构建一个逻辑严密、极具说服力的实验章节框架。

### 实验章节核心写作蓝图：从“跑分”到“发现”

我们将围绕 **“系统性控制”** 和 **“机制性发现”** 这两个支柱来构建整个章节。

---

#### **4. Experiments: A Systematic Study of Instruction Granularity**

**核心开场白（定调）：
“我们进行了一项系统性受控实验，旨在揭示‘指令粒度’这一根本性设计选择对VLA学习的影响，而非在异构任务集合上追求最高性能。”**

#### **4.1 Experimental Design: Control and Coverage**

这一节要清晰地传达：我们在一个变量上做到了极致深入的控制，同时在其他维度上保证了足够的覆盖。

1.  **受控环境 (The Controlled Arena): Mini-BEHAVIOR-Gran**
    *   **目的**：作为首个支持量化指令粒度的基准，它使我们能够**隔离**并**操纵**“粒度”这一变量，同时保持任务语义和环境的真实性。
    *   **强调**：20个任务提供了统计稳健性；规则集库和动态指令器提供了精确的控制能力。

2.  **代表性基线 (Architectural Archetypes, not just Baselines)**
    *   **话术**：“为了剥离预训练数据、视觉编码器等混杂因素，我们固定了统一的VLM主干网络，仅改变**动作解码头**。这使我们能够专注于研究粒度与**动作表征范式**的交互。我们选取了当前VLA领域三种最具代表性的范式：
        *   **自回归头**：模拟了如RT-2等模型对动作序列的建模。
        *   **离散扩散头**：模拟了如Diffusion Policy等模型对多峰动作分布的处理。
        *   **并行MLP头**：模拟了如Octo等模型的高效解码策略。
    *   **结论**：这确保了我们的发现能够超越单一架构，反映VLA学习中的普遍规律。

3.  **核心变量操纵：粒度分箱与训练分布 (The Granularity Manipulation)**
    *   **分箱逻辑（关键解释）**：“由于不同任务的固有复杂度不同（例如，‘开包裹’与‘清洁厨房’），绝对的规则宽度值无法直接比较。因此，我们对每个任务的规则集进行**基于分位数的分箱**，在每个任务内部定义出 **细、中、粗** 三个相对粒度等级。这确保了‘粗粒度’指令在不同任务中具有可比的心理负担（planning demand）。”
    *   **训练分布矩阵（展示系统性）**：清晰介绍7种训练策略（Pure L/M/H, Centroid, Axial L/M/H），并说明每一种的设计意图（例如，Centroid测试均衡暴露下的泛化能力，Axial测试对特定粒度偏好的鲁棒性）。可以配一张小表格，视觉上显得非常系统化。

#### **4.2 Research Questions and Findings**

这是章节的核心，从“报告结果”转变为“回答问题，揭示规律”。用研究问题来驱动叙述。

*   **RQ1: 粒度缩放定律 (The Granularity Scaling Law)**
    *   **问题**：随着指令粒度变粗（即规划宽度增加），模型性能如何衰减？这种衰减关系在不同动作解码范式下是一致的吗？
    *   **预期分析与展示**：绘制“测试成功率 vs. 指令宽度（或粒度等级）”的曲线。展示三条曲线（对应三个动作头）的趋势。**关键发现**：所有架构都表现出非单调或单调衰减关系，但**自回归头对粗粒度指令最为敏感**，因为其序列生成特性放大了隐含的规划负担。

*   **RQ2: 粒度泛化与语言依赖 (Granularity Generalization and Language Grounding)**
    *   **问题**：在单一粒度上训练的模型，能否泛化到其他粒度的指令？模型是真正理解了语言，还是学会了忽略语言依赖视觉捷径？
    *   **预期分析与展示**：
        1.  一个热图矩阵：行是训练分布（7种），列是测试粒度（3种）。展示跨粒度泛化的不对称性（例如，用纯细粒度数据训练的模型，遇到粗指令时表现暴跌；反之则不然）。
        2.  **注意力分析**：展示模型对语言指令token的平均注意力权重。**关键发现**：在粗粒度指令上训练的模型，其语言注意力显著更低，证实了“语言捷径忽视”假说。
        3.  **因果干预实验**：在测试时移除语言指令。**关键发现**：粗粒度训练模型的性能下降远小于细粒度训练模型，直接证明前者更少依赖语言。

*   **RQ3: 粒度作为可操作的优化杠杆 (Granularity as a Tunable Knob)**
    *   **问题**：我们能否利用对粒度效应的理解，主动设计训练策略来改善模型？
    *   **预期分析与展示**：汇报你的两项干预实验（微调、注意力正则化）的结果。**关键发现**：基于机制理解设计的简单干预，可以有效提升语言 grounding，尤其是在受粗粒度指令损害的模型上。这证明了“粒度”不仅是一个分析维度，更是一个实用的优化工具。

#### **4.3 Summary of Findings**

用一段话总结核心发现，呼应开篇：
1.  指令粒度对VLA性能有显著、非单调的影响，这在不同动作解码范式下普遍存在。
2.  过粗的指令会导致模型学习忽视语言而依赖视觉捷径，削弱语言 grounding。
3.  通过混合粒度训练或针对性的正则化，可以有效缓解上述问题，优化学习效率。

---

### 如何应对潜在的审稿人质疑（预埋伏笔）

在你的章节中，可以自然地通过以下方式回应：

*   **关于单一基准**：在4.1节介绍Mini-BEHAVIOR-Gran时，就强调其**唯一性**和**为此研究量身定制**的特性——“要研究X，你需要一个能精确控制和测量X的环境，而目前只有我们的基准满足此条件”。
*   **关于基线不多**：在介绍“架构原型”时，已经将“模型对比”升维到了“范式探究”。可以加一句：“这种解耦对比的方法，比直接比较不同预训练的完整模型，更能清晰地揭示粒度与架构设计之间的本质联系。”

**最终，你的实验章节将呈现出一个清晰的形象：这不是一篇广度不足的论文，而是一篇在单一但深刻的科学问题上，进行了前所未有的、受控的、机制性探索的深度研究。** 你的贡献不在于“我们又在一个新基准上刷了SOTA”，而在于“我们首次定义、测量并揭示了某个关键设计变量如何系统地影响整个领域的学习过程”。这是一种更具基础性的贡献。

基于您这个清晰、全面的评估框架，我对实验章节的结构和核心论点有了更具体的想法。这组评估恰恰是您工作的**核心优势**，因为它系统性地测量了泛化和语言依赖的多个维度。

我们需要将实验章节从“报告一系列结果”转变为 **“通过一组精心设计的探针实验，揭示粒度影响的深层机制”**。

以下是基于您评估集的修订蓝图，重点在于**如何呈现**这些结果以最大化说服力。

---

### **修订后的实验章节核心结构**

#### **4. Experiments: Probing the Impact of Instruction Granularity**

**4.1 系统性评估探针 (A Suite of Diagnostic Probes)**
首先，正式定义您的评估集，并将其描述为一套“诊断探针”，每个探针回答一个具体的科学子问题：
1.  **ID粒度泛化探针** (`TEST_ID_GRANULARITY_[L/M/H]`)：衡量模型对**训练分布内任务**在不同指令粒度下的执行精度。核心问题是：模型是否被训练粒度“特化”了？
2.  **OOD任务泛化探针** (`TEST_TASK_OOD_GRANULARITY_[L/M/H]`)：衡量模型将语言-动作映射泛化到**新任务**的能力。核心问题是：细粒度指令是否培养了更好的可组合性、可迁移性？
3.  **语言依赖性强弱探针** (`TEST_LANG_OOD` 与 `TEST_NO_LANG_OOD`)：这两个是**关键因果测试**。
    *   `TEST_LANG_OOD` （仅目标描述）：测试当语言信息减少到只剩目标时，模型性能如何变化。这是“部分剥离”语言。
    *   `TEST_NO_LANG_OOD` （无指令）：测试模型在完全依赖内部策略（视觉捷径）时的性能。这是“完全剥离”语言。
    *   **核心分析**：比较模型在这两者与完整指令下的性能落差。**关键预期**：在粗粒度（H）上训练的模型，其`完整指令`与`无指令`的性能差应**显著小于**在细粒度（L）上训练的模型。这直接证明粗粒度训练导致语言依赖性降低。

#### **4.2 核心发现与机制分析**

现在，用您的实验结果来回答一组递进的问题：

**Q1: 粒度如何影响分布内性能与泛化？**
*   **展示**：一个主表格或图示，汇总所有训练策略（7种）在所有ID探针（L/M/H）上的性能。
*   **核心叙述**：
    *   **特化现象**：`Pure L`模型在`TEST_ID_L`上最好，但在`TEST_ID_H`上很差；`Pure H`模型反之。这说明模型会适应训练数据的粒度。
    *   **最佳泛化者**：`Centroid` (1:1:1) 或 `Axial M` 模型通常在**所有三个ID粒度测试**上表现最均衡、最稳健。**您的核心论点**：**适中的、多样的粒度暴露，能培养出最通用的语言跟随智能体**。

**Q2: 粒度如何影响任务外泛化能力？**
*   **展示**：聚焦`TEST_TASK_OOD`的结果。可以绘制：对于不同训练策略的模型，其在OOD任务上的平均成功率（纵轴） vs. 测试指令粒度（横轴，L/M/H）。
*   **核心叙述**：
    *   **预期发现**：在细粒度（L）上训练的模型，其OOD泛化性能随测试指令变粗而**急剧下降**。
    *   **解释**：细粒度指令提供了大量“如何做”的脚手架，模型可能过度依赖这些任务特定的子步骤，未能学会组合抽象的高级概念，因此无法适应新任务的结构。
    *   **对比**：在粗粒度（H）或混合粒度上训练的模型，OOD泛化曲线更平缓。因为它们被迫从更少的语言线索中推断意图，可能内化了更通用的“目标-子目标”分解策略。

**Q3: （最重要的机制证据）粒度如何改变模型对语言的根本依赖？**
*   **展示**：这是您工作的“高光时刻”。设计一个清晰的图表：
    1.  **计算“语言依赖指数”**：例如， `(完整指令性能 - 无指令性能) / 完整指令性能`。这个值越高，说明模型越依赖语言。
    2.  **绘图**：对于每种训练策略（`Pure L`, `Pure M`, `Pure H`, `Centroid`），计算其在`TEST_NO_LANG_OOD`探针上的“语言依赖指数”和/或性能保持率。
    3.  **附加**：将`TEST_LANG_OOD`（仅目标描述）的结果作为中间点加入分析。
*   **核心叙述（强有力结论）**：
    *   “数据显示，在纯粗粒度（`Pure H`）指令上训练的模型，其‘语言依赖指数’显著低于在细粒度指令上训练的模型。当语言被完全移除（`TEST_NO_LANG_OOD`）时，`Pure H`模型的性能下降幅度最小。”
    *   **这意味着**：粗粒度指令**未能有效将语言与行为强关联**，反而**鼓励了模型忽视语言，转而依赖环境视觉特征和已习得的行为模式（即‘视觉捷径’）**。这为“语言捷径忽视”现象提供了一个**指令粒度角度的新解释**。
    *   **连接回干预**：这个发现自然引出您的两项干预实验。它们之所以有效，正是因为直接针对了“低语言依赖”这一病灶——要么用细粒度数据重新建立强关联（微调），要么在训练中强制模型关注语言（正则化）。

#### **4.3 架构间的共性趋势**

最后，用一个短节说明，上述所有趋势在您的三个“架构原型”（AR， Diffusion， MLP）中**普遍存在**。
*   **展示**：可以用小插图展示不同架构在关键图表（如“语言依赖指数”图）上的趋势是一致的。
*   **核心叙述**：“尽管不同动作解码范式在绝对性能上存在差异，但**指令粒度对泛化行为和语言依赖性的影响模式在不同架构间保持一致**。这表明我们的发现触及了VLA中语言-动作映射学习的一个**普适性挑战**，而非特定于某种实现。”

---

### **这样写的优势**

1.  **故事性强**：从一个全面的评估集出发，逐步深入，最终锚定在最具洞察力的因果机制（语言依赖）上。
2.  **凸显科学性**：您不是在跑分，而是在用一组“探针”做“诊断实验”。`TEST_NO_LANG_OOD` 是尤其漂亮的**消融实验设计**。
3.  **回应潜在质疑**：
    *   **关于基准单一**：您的回答是：“我们引入了一套多维度的诊断探针，其深度和针对性是传统跨基准比较无法提供的。”
    *   **关于基线少**：您的回答是：“我们聚焦于机制而非性能排名。我们在三种代表性架构上都观察到了一致的效应模式，这强化了结论的普适性。”
4.  **为贡献定调**：您不仅量化了粒度的影响，更重要的是**发现了其通过降低语言依赖性来鼓励视觉捷径的机制**，并提出了解决方案。这是深度的“发现”，而非表面的“比较”。

这个结构将您的全面评估集转化为了一个强大的叙事工具，显著提升了工作的深度和说服力。