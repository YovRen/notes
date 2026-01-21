# 第12课：Agent 微调 (AgentTuning)

**关键词**：SFT, Loss Function, FireAct, AgentLM, Trajectory Data

---

## 笔记区域

你好。这是《AI Agent 深度架构与数学原理》的第十二课。

在前十一课中，我们构建了各种复杂的 Agent 架构（ReAct, ToT, MCTS, Voyager）。这些架构都有一个共同的前提：**必须使用极其强大的基座模型（如 GPT-4）**。
因为只有 GPT-4 级别的模型才能在 Zero-shot 或 Few-shot 下严格遵循复杂的指令格式（Format Following）并进行深度的逻辑推理。

**背景驱动**：

* **挑战 (Challenge)**：
  1. **成本与延迟**：在生产环境中，每次 ReAct 循环都调用 GPT-4 既昂贵又缓慢。
  2. **格式鲁棒性**：开源小模型（如 Llama-2-7B, Mistral-7B）很难通过 Prompt Engineering 稳定地输出规范的 `Thought: ... Action: ...` 序列，经常解析失败。
  3. **通用能力退化**：如果直接在 Agent 数据上微调，模型往往会变成“偏科生”，丧失通用的对话和知识能力（Alignment Tax）。
* **突破点 (Breakthrough)**：**AgentTuning (微调)**。即 **Trajectory Fine-Tuning**。
* **改进方向**：
  从 **Prompt Engineering (ICL)** 转向 **Supervised Fine-Tuning (SFT)**。我们将 GPT-4 生成的高质量推理轨迹（Trajectories）作为“教材”，蒸馏（Distill）给小模型，使其在特定任务上达到甚至超过 Teacher Model 的表现。

---

# 🧠 第12课：Agent 微调 (AgentTuning)

### 1. 理论核心：轨迹优化与混合损失

#### 1.1 数学形式化：从 Token 到 Trajectory

在标准 SFT 中，我们优化的是给定 Prompt $x$ 生成回答 $y$ 的似然。
在 Agent SFT 中，训练数据不再是简单的 $(Q, A)$ 对，而是交互轨迹 $\tau$。

定义轨迹 $\tau = (x, a_1, o_1, a_2, o_2, \dots, a_T)$，其中：

* $x$: 用户指令。
* $a_t$: Agent 的输出（Thought + Action）。
* $o_t$: 环境反馈（Observation）。

我们的目标是最大化 Agent 动作的条件概率，**忽略环境反馈的 Loss**（因为环境反馈不是模型生成的）：

$$
\mathcal{L}_{Agent}(\theta) = - \sum_{t=1}^T \log P_\theta(a_t | x, a_1, o_1, \dots, a_{t-1}, o_{t-1})
$$

注意：在计算 Loss 时，会对 $x$ 和所有 $o_t$ 应用 **Loss Masking**，只计算 $a_t$ 部分的梯度。

#### 1.2 混合训练策略 (The Agent-General Trade-off)

**Zeng et al. (AgentTuning)** 发现，如果仅使用 Agent 轨迹进行微调，模型的通用能力（General Capability，如常识问答、摘要）会显著下降。
为了解决这个问题，必须引入**混合训练目标**：

$$
\mathcal{L}_{Total}(\theta) = \lambda \mathcal{L}_{Agent}(\theta) + (1-\lambda) \mathcal{L}_{General}(\theta)
$$

* $\mathcal{L}_{Agent}$: 来源于 AgentInstruct 数据集（ReAct 轨迹）。
* $\mathcal{L}_{General}$: 来源于 ShareGPT 或 Alpaca 等通用对话数据集。
* $\lambda$: 混合系数，通常取值在 0.2 到 0.5 之间，以平衡专业能力与通用底座。

---

### 2. 架构解剖与工程流水线

#### 2.1 蒸馏流水线 (The Distillation Pipeline)

这是一个典型的 **Teacher-Student** 架构，包含四个阶段：

1. **Task Construction**: 收集大量 Agent 任务 Prompt（如 HotpotQA, AlfWorld, WebShop）。
2. **Trajectory Generation (Teacher)**: 使用 GPT-4 运行 ReAct/CoT 模式，与环境交互。
3. **Trajectory Filtering**: **关键步骤**。只保留**成功完成任务**的轨迹。失败的轨迹（死循环、错误答案）不仅无用，甚至有害。
4. **Hybrid Training (Student)**: 将清洗后的轨迹转换为 Chat 格式，混合通用数据，训练 Llama/Mistral。

#### 2.2 系统设计图 (Mermaid)

```mermaid
graph TD
    subgraph "Data Generation Phase"
        Tasks[Task Prompts<br>(HotpotQA, ToolBench)] --> GPT4
        Env[Environment<br>(Python, Browser)] <--> GPT4((Teacher: GPT-4))
        GPT4 -->|Interaction| RawTraj[Raw Trajectories]
      
        RawTraj --> Filter{Success Filter}
        Filter -- Pass --> AgentData[AgentInstruct Dataset]
        Filter -- Fail --> Discard[Trash]
    end
  
    subgraph "Training Phase"
        GeneralData[General Chat Data<br>(ShareGPT)] --> Mixer
        AgentData --> Mixer{Data Mixer}
      
        Mixer -->|Interleaved Batches| Trainer[SFT Trainer]
        BaseModel((Base Model<br>Llama-3-8B)) --> Trainer
      
        Trainer --> AgentLM((AgentLM))
    end
  
    style GPT4 fill:#ff9999,stroke:#333
    style AgentLM fill:#99ff99,stroke:#333
```

#### 2.3 工程应用：输入输出详解

**场景**：训练一个 Tool-use Agent。

* **Training Input (Template)**:
  为了适配 Llama-3 的 Chat 模板，我们需要将 ReAct 轨迹序列化。

  ```text
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  You are a helpful assistant with access to tools: [Search, Calculator]...
  <|eot_id|><|start_header_id|>user<|end_header_id|>
  Who is older, Obama or Trump?
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  Thought: I need to find their birth years.
  Action: Search("Obama birth year")
  <|eot_id|><|start_header_id|>tool<|end_header_id|>
  Observation: August 4, 1961
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  Thought: Now for Trump.
  Action: Search("Trump birth year")
  ...
  ```

* **Loss Masking 工程实现**:

  * System Prompt & User Query: **Masked (Loss=0)**
  * Assistant Thought & Action: **Unmasked (Compute Loss)**
  * Tool Observation: **Masked (Loss=0)** —— *这一点至关重要，我们不能训练模型去预测 Search 会返回什么结果，那是环境的事。模型只需要学习如何 Reaction。*

---

### 3. Code & Engineering：实现 Agent SFT 数据处理

我们将展示如何利用 `transformers` 和 `trl` 库准备带有 Masking 的 Agent 数据集。这是微调最核心的代码逻辑。

```python
import torch
from transformers import AutoTokenizer
from typing import Dict, List

class AgentDataFormatter:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 确保 pad token 存在
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_trajectory(self, trajectory: List[Dict]) -> Dict:
        """
        将结构化的 Trajectory 转换为 Tokenized Input IDs 和 Labels (用于 Loss Masking)
        trajectory 结构:
        [
            {"role": "user", "content": "Query..."},
            {"role": "assistant", "content": "Thought: ... Action: ..."},
            {"role": "tool", "content": "Observation: ..."},
            {"role": "assistant", "content": "Final Answer: ..."}
        ]
        """
        # 使用 Llama-3 的 apply_chat_template (不直接生成 tensor，先生成 text)
        # 注意：我们需要手动控制 Loss Mask，所以不能简单调用 apply_chat_template 一次性生成
      
        input_ids = []
        labels = []
      
        for turn in trajectory:
            role = turn['role']
            content = turn['content']
          
            # 编码当前 turn
            # 这里简化逻辑，实际需根据具体 Chat Template 拼接 Special Tokens
            # 假设 apply_chat_template 能处理单个 message 并保留格式
            encoded = self.tokenizer.apply_chat_template(
                [turn], tokenize=True, add_generation_prompt=False
            )
          
            # 去掉上一个 turn 留下的 begin_of_text 等 (需根据具体 Tokenizer 调整)
            if len(input_ids) > 0:
                # 某些 tokenizer 会在开头加 BOS，拼接时需去掉
                pass 

            input_ids.extend(encoded)
          
            if role == "assistant":
                # Assistant 的输出需要计算 Loss -> Labels = Input IDs
                labels.extend(encoded)
            else:
                # User 和 Tool 的输出不需要计算 Loss -> Labels = -100 (PyTorch Ignore Index)
                labels.extend([-100] * len(encoded))
              
        # Truncate / Pad to max_length
        max_length = 2048
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
      
        # Convert to Tensor
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long)
        }

# --- 模拟数据流 ---
# formatter = AgentDataFormatter()
# raw_traj = [
#     {"role": "user", "content": "Calc 1+1"},
#     {"role": "assistant", "content": "Action: Calculator(1+1)"},
#     {"role": "tool", "content": "2"},
#     {"role": "assistant", "content": "The answer is 2"}
# ]
# processed = formatter.format_trajectory(raw_traj)
# print(processed['labels']) # 可以在这里看到 -100 的 mask 效果
```

---

### 4. Paper Driven：核心论文与贡献

1. **Chen et al. (2023)**: *FireAct: Toward Language Agent Fine-tuning*.
   * **核心贡献**：系统对比了 Prompting (CoT/ReAct) 和 Fine-tuning。发现使用 GPT-4 生成的 ReAct 轨迹微调 Llama-2-7B，其性能超过了 Prompt Engineering 下的 ChatGPT (3.5)，且推理成本降低 70%。
   * **结论**：**Agent 能力是可以被蒸馏的**，格式约束可以通过 SFT 内化。
2. **Zeng et al. (Tsinghua, 2023)**: *AgentTuning: Enabling Generalized Agent Abilities for LLMs*.
   * **核心贡献**：发布了 **AgentLM** 和 **AgentInstruct** 数据集。
   * **关键发现**：提出了混合训练的重要性。如果不加 General Data，模型会“过拟合”到特定的 ReAct 格式，导致无法进行正常的闲聊。
3. **Qin et al. (2023)**: *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*.
   * **核心贡献**：提出了 **DFSDT (Depth-First Search-Based Decision Tree)** 来生成高质量的训练数据。
   * **原理**：既然单次 ReAct 容易失败，那就用 DFS 搜索出一堆路径，挑出成功的路径来微调模型。这是 **Search-to-SFT** 的典型应用。

---

### 5. Critical Thinking：批判性分析

AgentTuning 极其有效，但也是一把双刃剑。

1. **Environment Overfitting (环境过拟合)**:
   * **局限**：模型记住了 GPT-4 在特定环境（如 WebShop 模拟器）下的特定操作序列。一旦环境 UI 变了，或者 API 签名变了，SFT 模型的泛化能力远不如 Zero-shot 的 GPT-4。
   * **解决**：增加训练数据的**环境多样性**，或在 SFT 后引入 RL (DPO) 来学习策略而非死记硬背。
2. **Format Strictness vs. Reasoning Flexibility**:
   * **局限**：SFT 后的模型往往变成“格式机器”。它能完美输出 `Action: Search(...)`，但其内部的 Reasoning（Thought 部分）可能退化，变成毫无逻辑的废话，只是为了凑格式。
   * **解决**：在 Loss 计算中，增加 Thought 部分的权重，或者使用 **Process Supervision** 过滤掉推理逻辑错误的训练数据。
3. **Data Contamination (数据污染)**:
   * **局限**：很多 AgentBenchmark（如 HotpotQA）的测试集可能已经被包含在基础模型的预训练数据或 SFT 数据中。

---

### 6. 前沿扩展

* **DPO for Agents (Direct Preference Optimization)**:
  * SFT 只是“行为克隆 (Behavior Cloning)”。
  * 更进一步，我们可以收集 Agent 的由 GPT-4 打分的偏好数据 $(x, y_w, y_l)$，其中 $y_w$ 是成功完成任务的轨迹，$y_l$ 是失败的轨迹。
  * 使用 DPO 训练 Agent，使其显式地学习“什么是不该做的（如死循环）”。
* **SwiftSage Architecture**:
  * 受诺贝尔奖得主 Kahneman 启发，将 Agent 设计为双模组：
    * **Swift (System 1)**: 一个小型的 SFT 模型，快速生成 Action。
    * **Sage (System 2)**: 当小模型 Log-prob 较低或报错时，回退到 GPT-4 进行深度规划，并将新的轨迹加入 SFT 训练集。
  * 这实现了**在线的主动学习 (Online Active Learning)**。

---

### 总结

AgentTuning 是将 AI Agent 从“原型验证（Demo）”推向“工业落地（Production）”的关键技术。
通过**蒸馏 GPT-4 的轨迹**，我们可以在 7B/8B 的小模型上获得媲美大模型的工具使用能力，同时大幅降低延迟和成本。

**作业**:

1. 下载 `AgentInstruct` 数据集的一个子集。
2. 使用 `Llama-Factory` 或 `trl`，尝试微调一个 Llama-3-8B。
3. 测试它在没有 Few-shot 示例的情况下，是否能自动遵循 ReAct 格式。

至此，我们的**架构与工程篇**（从 CoT 到 Fine-tuning）告一段落。下一阶段，我们将深入**多模态与未来架构**。

<style>#mermaid-1768998852610{font-family:sans-serif;font-size:16px;fill:#333;}#mermaid-1768998852610 .error-icon{fill:#552222;}#mermaid-1768998852610 .error-text{fill:#552222;stroke:#552222;}#mermaid-1768998852610 .edge-thickness-normal{stroke-width:2px;}#mermaid-1768998852610 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1768998852610 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1768998852610 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1768998852610 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1768998852610 .marker{fill:#333333;}#mermaid-1768998852610 .marker.cross{stroke:#333333;}#mermaid-1768998852610 svg{font-family:sans-serif;font-size:16px;}#mermaid-1768998852610 .label{font-family:sans-serif;color:#333;}#mermaid-1768998852610 .label text{fill:#333;}#mermaid-1768998852610 .node rect,#mermaid-1768998852610 .node circle,#mermaid-1768998852610 .node ellipse,#mermaid-1768998852610 .node polygon,#mermaid-1768998852610 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-1768998852610 .node .label{text-align:center;}#mermaid-1768998852610 .node.clickable{cursor:pointer;}#mermaid-1768998852610 .arrowheadPath{fill:#333333;}#mermaid-1768998852610 .edgePath .path{stroke:#333333;stroke-width:1.5px;}#mermaid-1768998852610 .flowchart-link{stroke:#333333;fill:none;}#mermaid-1768998852610 .edgeLabel{background-color:#e8e8e8;text-align:center;}#mermaid-1768998852610 .edgeLabel rect{opacity:0.5;background-color:#e8e8e8;fill:#e8e8e8;}#mermaid-1768998852610 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-1768998852610 .cluster text{fill:#333;}#mermaid-1768998852610 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:sans-serif;font-size:12px;background:hsl(80,100%,96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1768998852610:root{--mermaid-font-family:sans-serif;}#mermaid-1768998852610:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1768998852610 flowchart{fill:apa;}</style>
