## 1. 引言（\S 1）：重新组织，突出“缺乏工具”这个gap

**当前问题**：直接跳到“缺乏统一度量”，没有先说明“为什么这个问题重要”和“为什么没人研究”。

**修改后结构**：

- **第1段：问题的重要性**

  > 指令粒度是语言引导的具身智能中一个核心变量。对于最近很火的VLA模型，实际上他们的训练方式就是multimodality alignment: VLA training (often via fine-tuning) aligns these pre-existing visual/language features with new action tokens or continuous action trajectories. Thus, 指令的粗细程度直接影响多模态对齐的难度，因此必然带来不同的agent behavior。理解这种影响对于构建鲁棒的、能适应人类多样指令的agent至关重要。

- **第2段：研究gap——为什么没人深入研究这个问题？**

  > 然而，指令粒度对VLA agents行为的影响至今仍是一个**under-explored**的问题。根本原因在于缺乏必要的研究工具：
  >
  > - **缺乏受控benchmark**：现有benchmark（ALFRED, BEHAVIOR等）的指令都是**静态的**——每个任务只提供一种粒度的指令。这使得我们无法进行“同一任务下不同粒度指令”的对比研究，无法分离粒度变量。
  > - **缺乏公认的量化指标**：即使想研究，我们也没有一个well-accepted metric来定义和量化“粒度”。token count、entity count等文本指标是否有效？是否有一个更本质的、能反映决策负担的指标？这些问题尚无答案。

- **第3段：我们的解决方案——提供工具**

  > 为了填补这一空白，我们构建了**`Mini-BEHAVIOR-Gran`**——第一个支持同一任务多粒度指令的benchmark。我们从细粒度规则出发，通过规则合并生成粒度递增的指令集合，从而实现了对指令粒度的系统控制。
  >
  > 在这个benchmark上，我们提供了多种粒度量化指标：文本层面的token count、entity count、action count，以及从规划领域引入的**rule width**——一个衡量指令隐含决策负担的形式化指标。

- **第4段：利用工具发现新知识**

  > 利用这个benchmark，我们首先评估了这些粒度指标的有效性，发现**width是与agent performance最稳定相关的指标**，优于文本启发式指标。基于width这一可靠度量，我们系统研究了不同粒度训练对VLA agents的影响，揭示了一个非单调的**U形性能曲线**，并诊断出其背后的机制——**浅层接地（shallow grounding）**，即粗粒度指令的高可预测性（P(l|v)）诱导agent走视觉捷径。我们还发现了**非对称泛化**现象和**宽度容量上限**（w≥4）。

- **第5段：贡献列表**

  > 1. **新benchmark**：我们发布了`Mini-BEHAVIOR-Gran`，第一个支持同一任务多粒度指令受控研究的benchmark，为社区提供了研究指令粒度问题的工具。
  > 2. **指标评估**：我们在这个benchmark上系统比较了多种粒度量化指标，发现**rule width**是与agent performance最consistent的指标，并论证了其跨任务的有效性。
  > 3. **新发现**：基于width，我们首次揭示了粒度与性能之间的U形关系，并诊断出其背后的浅层接地机制，为训练鲁棒VLA agents提供了关键洞见。



### Section 2: Related Work

**目标**：指出现有研究缺乏对指令粒度的受控研究，缺乏统一度量和合适 benchmark。

- **2.1 Instruction Granularity in Language-Guided AI**
  回顾自然语言处理中基于文本特征的粒度度量（如句子长度、实体数）和基于任务分解的粒度概念（如子目标分解深度）。指出前者无法反映决策负担，后者无法跨任务比较。
- **2.2 Embodied AI Benchmarks for Instruction Following**
  综述 ALFRED、BEHAVIOR、Mini-BEHAVIOR 等基准。核心论点：这些基准的指令是静态的，每个任务只提供一种粒度的指令，无法支持“同一任务不同粒度”的对比研究。这解释了为什么指令粒度的影响至今未被系统探究。
- **2.3 Width‑Based Planning**
  简要介绍宽度规划的基本思想，为后续引入规则宽度（rule width）做理论铺垫。强调宽度衡量的是状态空间中必须同时跟踪的特征数，与任务具体结构无关，因此具备跨任务比较的潜力。

### Section 3: Preliminaries: The Mini-BEHAVIOR Environment

**目标**：为 Benchmark 的构造提供必要的基础环境描述，使 Section 4 的构建过程自包含。

- **3.1 Task and Feature Representation**
  描述 Mini-BEHAVIOR 的设定：基于网格的 3D 环境、20 个长时程任务、共享的特征集合 Φ（包括物体状态、空间关系等）。给出特征的形式化定义（布尔型/数值型）。
- **3.2 Action Space and Dynamics**
  简要说明智能体的离散动作空间（移动、交互等）和状态转移规则，强调环境的内在 dynamics 决定了特征之间的依赖关系。
- **3.3 Why Mini-BEHAVIOR?**
  论证选择 Mini-BEHAVIOR 作为基础的原因：它保留了长时程、异质性的任务结构（使粒度调节有意义），同时简化了感知与控制（避免混淆因素），是研究指令粒度的理想实验床。

### Section 4: Mini-BEHAVIOR-Gran Benchmark

**目标**：详细呈现 **贡献1**——首个支持同一任务多粒度指令的 benchmark，并附带多种粒度量化指标。

- **4.1 Constructing Multi‑Granularity Instructions**
  - **起点**：使用宽度规划器（BFWS）为每个任务生成最优的、最细粒度的规则序列（每一步仅改变 1-2 个特征）。
  - **规则合并**：定义合并操作，将相邻规则迭代合并，得到一系列宽度递增的规则集（从 w=0 直到无法合并）。强调合并过程独立于宽度定义，只是工程手段。
  - **自然语言生成**：通过模板将规则集转化为自然语言指令，确保同一任务的不同指令仅在粒度上有系统差异。
- **4.2 Granularity Metrics**
  - **文本层面指标**：token count, entity count, action verb count。
  - **规则宽度（Rule Width）**：形式化定义单条规则宽度 w(r) 和指令宽度 w(R) = max w(r_i)。解释其直观含义——执行指令时必须同时跟踪的最大特征数，并举例说明。
  - **为什么 decomposition depth 不行**：因为分解深度依赖于任务自身的子目标结构，无法跨任务公平比较；而宽度基于共享特征集合 Φ，具备跨任务可比性。
- **4.3 Benchmark Statistics and Oracle Instructor**
  - 数据集规模：20 个任务，每个任务最多 6 个宽度级别（w=0,1,2,3,4,≥5），每个级别 50 条训练轨迹 + 10 条测试轨迹。
  - Oracle Instructor 机制：在推理时根据当前状态选择可用规则，并生成相应指令，确保性能差异仅来源于接地能力。
  - 公开资源：代码、数据、指标计算脚本均开源。

### Section 5: Experiments

**目标**：首先验证各粒度指标的有效性（**贡献2**），然后基于最好的指标（宽度）深入分析粒度对 VLA 行为的影响（**贡献3**）。

- **5.1 Experimental Setup**
  介绍模型架构（三种动作解码策略）、训练设置（Pure F/M/C、Centroid、Axial 等）、评估指标（成功率 SR）。
- **5.2 Which Metric Best Captures Granularity?**
  - **方法**：在 Centroid 训练（保证各粒度均衡暴露）下，计算各指标（token count, entity count, action verb count, width）与 SR 的斯皮尔曼相关系数，并控制任务 horizon 作为协变量。
  - **结果**：宽度是唯一在所有任务和 horizon 下均保持显著负相关的指标；文本指标波动大，甚至在某些情况下方向相反。
  - **结论**：宽度是最 consistent 的粒度量化指标，因此后续分析基于宽度展开。 ← **贡献2**
- **5.3 The U‑Shaped Performance Curve**
  - **设置**：Pure F/M/C 训练（只接触单一粒度类别）。
  - **发现**：SR 随宽度呈现非单调 U 形——w=0 高，w=1 低，w≥2 回升，且 w≥4 时所有模型性能崩溃（容量上限）。
  - **可视化**：绘制学习曲线（Fig. 4），展示不同动作解码架构下的 U 形。
- **5.4 Diagnosing the Rebound: Shallow Grounding**
  - **假设**：粗粒度（w≥2）的高性能源于指令的高可预测性 P(l|v)，诱导智能体发展视觉捷径，而非真正理解语言。
  - **实验 1**：训练一个 VLM 预测器，计算指令预测的交叉熵损失，发现损失随宽度增加而降低（P(l|v) 增加）。
  - **实验 2**：语言消融——将完整指令替换为静态目标描述，测量 SR 下降幅度 ΔSR。发现粗粒度训练模型的 ΔSR 显著小于细粒度模型，表明其对语言缺失不敏感。
  - **结论**：粗粒度性能回升是浅层接地（视觉捷径）所致。
- **5.5 Implications for Training: Mixed Granularity and Asymmetric Transfer**
  - **混合粒度训练**：比较 Pure 训练与 Axial/Centroid 训练，发现混合粒度（尤其是 Axial-F）能显著增大 ΔSR，即加深语言接地。
  - **非对称泛化**：跨粒度迁移表（Table 1）显示，粗→细迁移成功，但细→粗失败。这一模式与浅层接地假说一致。
  - **讨论**：为训练鲁棒的 VLA 智能体提供启示——必须暴露于多种粒度，尤其是细粒度指令以强制语言依赖。

### Section 6: Conclusion

**目标**：总结三大贡献，并展望未来方向。

- 我们构建了 **`Mini-BEHAVIOR-Gran`**——首个支持同一任务多粒度指令的 benchmark，并提供了多种粒度量化指标。
- 通过系统性比较，我们发现 **rule width** 是与智能体性能最 consistent 的粒度度量，优于传统文本指标。
- 基于宽度，我们揭示了粒度与性能之间的 **U 形关系**，并诊断出粗粒度下的 **浅层接地机制**，为训练语言鲁棒的 VLA 智能体提供了关键洞见。
- 未来工作包括：将研究拓展至连续控制环境、探索动态指令粒度、结合显式记忆模块突破 w≥4 的容量上限。

### Section 7: Limitations

**目标**：坦诚讨论局限性，化解潜在审稿质疑。

- 离散动作空间的局限，未来需验证连续控制场景。
- Oracle Instructor 的理想化设定，但即使给定规则，w≥4 的失败仍证明接地瓶颈存在。
- 实验基于网格环境，但视觉简化反而提供了 grounding 难度的下界，真实环境只会更难。