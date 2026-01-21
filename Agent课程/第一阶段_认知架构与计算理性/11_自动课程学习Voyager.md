# 第11课：自动课程学习 (Voyager)

**关键词**：Curriculum Learning, Skill Library, 向量化存储, 代码自我验证, Lifelong Learning

---

## 笔记区域

你好。这是《AI Agent 深度架构与数学原理》的第十一课。

在前面的课程中（ReAct, Reflexion, MCTS, HuggingGPT），我们讨论的 Agent 都有一个共同点：**任务是给定的（Goal-Conditioned）**。用户输入一个目标，Agent 去执行，执行完（或失败）后，Agent 就“死”了，记忆也被重置（除非有 RAG）。
这种 Agent 缺乏**终身学习（Lifelong Learning）**的能力。

**背景驱动**：

* **挑战 (Challenge)**：
  1. **灾难性遗忘 (Catastrophic Forgetting)**：Agent 解决了一个难题（比如“制作一把铁镐”），但在下一次任务中，它完全忘记了怎么做，必须重新推理一遍。
  2. **探索效率低下**：在开放世界（如 Minecraft, 操作系统）中，Agent 不知道该干什么。如果没有明确目标，它就会原地发呆或随机游走。
* **突破点 (Breakthrough)**：**Voyager (Wang et al., 2023)**。它引入了三个核心概念：**自动课程（Automatic Curriculum）**、**迭代提示机制（Iterative Prompting）**和**技能库（Skill Library）**。
* **改进方向**：
  从 **Gradient-based Learning**（微调模型参数）转向 **Code-based Learning**（积累可执行的代码片段作为技能）。

---

# 🧠 第11课：自动课程学习 (Voyager)

### 1. 理论核心：最近发展区与技能冻结

#### 1.1 数学定义：自动课程 (Automatic Curriculum)

课程学习（Curriculum Learning）的核心是寻找一个任务序列 $\mathcal{T} = \{t_1, t_2, \dots, t_n\}$，使得 Agent 在时刻 $k$ 学习任务 $t_k$ 时，能够获得最大的**信息增益 (Information Gain)**。

在 Voyager 中，这被形式化为寻找**最近发展区 (Zone of Proximal Development, ZPD)**。
给定当前 Agent 的状态 $S_t$（背包物品、周围环境）和已掌握的技能集 $\Pi_{skill}$，下一个最优任务 $t_{next}$ 应该满足：

$$
t_{next} = \arg\max_{t} \left( P(\text{success} | S_t, \Pi_{skill}, t) \cdot V(t) \right)
$$

* $P(\text{success} | \dots)$: 成功的概率。任务不能太难（如没有木头就想造钻石剑）。
* $V(t)$: 任务的新颖性或价值。任务不能太简单（如反复挖泥土）。

#### 1.2 技能作为参数 (Code as Policies)

传统的 RL 将策略存储在神经网络权重 $\theta$ 中。Voyager 将策略存储为**代码片段 (Python/JavaScript Functions)**。
定义技能库 $\mathcal{L} = \{ (k_i, c_i) \}_{i=1}^N$，其中：

* $k_i$: 技能的 Embedding（Docstring 的向量表示）。
* $c_i$: 可执行的代码函数体。

当面对新任务 $t_{new}$ 时，策略 $\pi$ 变为 RAG 过程：

$$
\pi(a|s) \leftarrow \text{LLM}(\text{Prompt} + \text{Retrieve}(\mathcal{L}, t_{new}))
$$

这是一种**非参数化（Non-parametric）**的学习方式，避免了参数更新带来的遗忘问题。

---

### 2. 架构解剖与工程应用

#### 2.1 三大核心组件

Voyager 的架构是一个无限循环的**探索-学习-固化**过程：

1. **Automatic Curriculum (AC)**:
   * **Input**: 当前状态（Inventory, Biome）、完成的任务历史。
   * **Output**: 下一个目标（Task）。例如："Mine 1 wood log"。
2. **Iterative Prompting Mechanism (IPM)**:
   * **Input**: Task, Retrieved Skills, Environment Feedback (Error Trace)。
   * **Process**: 写代码 -> 运行 -> 报错 -> Self-Correction -> 成功。
   * **Output**: 成功的代码。
3. **Skill Library (SL)**:
   * **Action**: 将成功的代码清理、注释、向量化存储。
   * **Effect**: 技能被**冻结**。以后再需要 "Mine wood" 时，直接调用函数，不再经过 LLM 推理。

#### 2.2 系统架构图 (Mermaid)

```mermaid
graph TD
    subgraph "Voyager Loop"
        State[State: Inventory, Biome] --> AC[Automatic Curriculum]
        AC -->|Propose Task| IPM[Iterative Prompting (Coding Agent)]
    
        subgraph "Skill System"
            DB[(Vector DB: Skill Library)]
            DB -->|Retrieve Relevant Skills| IPM
            IPM -->|Execution Feedback| Env[Minecraft Env]
            Env -->|Success/Fail| IPM
        
            IPM -->|Success| Verify{Verification}
            Verify -->|Save Code| DB
        end
    
        Verify -->|Update State| State
    end
  
    style DB fill:#ff9999,stroke:#333
    style AC fill:#99ff99,stroke:#333
```

#### 2.3 工程应用：输入输出流

**场景**：Minecraft 初生状态。

1. **Input (To Curriculum)**:
   * State: "Time: Morning. Inventory: Empty. Nearby: Tree, Dirt. Biome: Plains."
   * Prompt: "Propose the next logical task to advance capabilities."
2. **Curriculum Output**:
   * Task: "Collect 3 wood logs." (因为没有木头无法做工具，这是最优前驱任务)。
3. **Input (To IPM)**:
   * Task: "Collect 3 wood logs."
   * Retrieved Skills: Empty (Initial).
4. **IPM Execution**:
   * GPT-4 Writes: `bot.dig(tree)`
   * Env Feedback: `Error: Target out of reach.`
   * GPT-4 Refines: `bot.pathfinder.goto(tree); bot.dig(tree)`
   * Env Feedback: `Success! Inventory: 3 wood logs.`
5. **Skill Storage**:
   * Function: `def mine_wood(): ...`
   * Description: "Navigate to the nearest tree and collect logs."
   * **Vector**: Embedding(Description).
6. **Next Loop**:
   * Curriculum see "3 wood logs". Next Task: "Craft a crafting table."

---

### 3. Code & Engineering：实现简易版 Voyager

我们将实现 Voyager 的核心逻辑：**课程生成**与**技能检索**。这里的关键是 Prompt Engineering 如何引导 LLM 进行“探索性规划”。

```python
import openai
from typing import List, Dict
import numpy as np

# 模拟向量数据库
class SkillLibrary:
    def __init__(self):
        self.skills: Dict[str, str] = {} # name -> code
        self.descriptions: List[str] = []
        self.vectors: List[np.ndarray] = []
  
    def add_skill(self, name: str, code: str, description: str):
        self.skills[name] = code
        self.descriptions.append(description)
        # Mock embedding
        self.vectors.append(np.random.rand(768)) 
        print(f"📚 Skill '{name}' added to library.")

    def retrieve(self, task_query: str, k=3) -> List[str]:
        if not self.skills:
            return []
        # Mock retrieval: 在实际中使用 cosine similarity
        print(f"🔍 Retrieving skills for: {task_query}")
        return list(self.skills.values())[:k]

class AutomaticCurriculum:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.completed_tasks = []

    def propose_next_task(self, agent_state: str) -> str:
        """
        核心逻辑：根据当前状态，利用 LLM 的常识图谱，寻找 ZPD (Zone of Proximal Development)
        """
        prompt = f"""
        You are a smart adventurer. 
        Current State: {agent_state}
        Completed Tasks: {self.completed_tasks}
    
        Reasoning rules:
        1. Propose a task that is difficult enough to be interesting but easy enough to be possible.
        2. Do not propose tasks that require items not in inventory (unless obtaining them is the task).
        3. Think about the tech tree hierarchy.
    
        Next Task:
        """
        # response = self.llm.chat.completions.create(...)
        # Mock response based on state
        if "Empty" in agent_state:
            return "Gather 3 Wood Logs"
        elif "Wood Logs" in agent_state:
            return "Craft a Crafting Table"
        return "Explore the world"

class IterativePrompter:
    def __init__(self, llm_client, skill_lib: SkillLibrary):
        self.llm = llm_client
        self.skill_lib = skill_lib

    def execute_task(self, task: str) -> bool:
        # 1. Retrieve useful skills
        context_skills = self.skill_lib.retrieve(task)
    
        # 2. Write Code (The "Action")
        code = self._generate_code(task, context_skills)
    
        # 3. Simulate Execution Environment
        success, feedback = self._simulate_env(code)
    
        # 4. Self-Correction Loop (Reflexion)
        retries = 0
        while not success and retries < 3:
            print(f"❌ Failed: {feedback}. Refining code...")
            code = self._refine_code(code, feedback)
            success, feedback = self._simulate_env(code)
            retries += 1
        
        if success:
            print(f"✅ Task '{task}' completed!")
            # 5. Extract reusable function and save
            func_name = task.lower().replace(" ", "_")
            self.skill_lib.add_skill(func_name, code, f"Skill to {task}")
            return True
        return False

    def _generate_code(self, task, skills):
        return f"def {task.replace(' ', '_')}(): pass # impl"
    
    def _simulate_env(self, code):
        # Mock environment feedback
        return True, "Execution Successful"

    def _refine_code(self, code, error):
        return code + " # fixed"

# --- Main Voyager Loop ---
# curriculum = AutomaticCurriculum(client)
# prompter = IterativePrompter(client, SkillLibrary())
# 
# state = "Inventory: Empty"
# while True:
#     task = curriculum.propose_next_task(state)
#     print(f"🎯 New Goal: {task}")
#     success = prompter.execute_task(task)
#     if success:
#         curriculum.completed_tasks.append(task)
#         state = "Inventory: Wood Logs" # State update simulation
#     else:
#         break
```

---

### 4. Paper Driven：核心论文与贡献

1. **Wang et al. (NVIDIA, 2023)**: *Voyager: An Open-Ended Embodied Agent with Large Language Models*.
   * **核心贡献**：首次展示了 LLM Agent 在没有梯度更新的情况下，通过**代码库积累**实现终身学习。在 Minecraft 中解锁的科技树里程碑是传统 RL 方法的 3.3 倍。
   * **关键机制**：利用 GPT-4 的编码能力（Coding）替代了传统的动作预测（Action Prediction）。代码具有**组合性（Compositionality）**和**抽象性（Abstraction）**，比原子动作更适合长程任务。
2. **Zhu et al. (2023)**: *Ghost in the Minecraft (GITM)*.
   * **对比**：GITM 侧重于分层规划（Hierarchical Planning），类似于我们第10课讲的 HuggingGPT，但应用于 Minecraft。Voyager 侧重于“无监督探索”和“技能发现”。
3. **Significant-Gravitas (2023)**: *AutoGPT*.
   * **对比**：AutoGPT 是 Goal-Oriented 的，给一个终极目标，它拆解执行。Voyager 是 Open-Ended 的，它自己给自己提目标。这是 **Curriculum Learning** 的本质区别。

---

### 5. Critical Thinking：批判性分析

Voyager 是 AI Agent 领域的一个里程碑，但它依然有局限：

1. **Code Hallucination & Environment Drift**:
   * **问题**：GPT-4 写的代码可能包含不存在的 API（幻觉）。或者，存入技能库的代码在游戏版本更新后失效（环境漂移）。
   * **解决**：需要一个 **Linter/Compiler** 作为 Evaluator（ReAct 思想）。对于环境漂移，需要引入 **Skill Maintenance** 机制，定期重新验证技能库的有效性。
2. **Context Explosion (Again)**:
   * **问题**：随着技能库增长，RAG 检索出的 Top-K 技能可能包含大量冗余代码，撑爆 Context。
   * **解决**：**Skill Pruning (技能剪枝)**。合并相似技能，或者对技能代码进行重构（Refactoring），只保留函数签名和 Docstring，需要执行时再展开。
3. **Cost (成本)**:
   * **问题**：Voyager 的每一个循环都涉及大量 GPT-4 调用（课程生成、代码编写、错误修正）。玩一小时 Minecraft 可能消耗几十美元。
   * **解决**：**Model Distillation**。用 GPT-4 探索出的技能轨迹去微调一个小模型（如 StarCoder），让小模型学会写 Minecraft 代码。

---

### 6. 前沿扩展

* **Voyager for Software Engineering (SWE-Agent)**:
  * 将 Minecraft 环境换成 **Linux Shell** 或 **GitHub Repo**。
  * Curriculum: "Fix bug #123" -> "Run Tests" -> "Refactor Module".
  * Skill Library: 积累常用的 Git 操作、Regex 处理、API 调用脚本。
  * 这是目前 Devin 等 AI 程序员背后的核心逻辑之一。
* **Generalist Agent**:
  * Voyager 证明了 **"Code as Policy"** 的可行性。未来的通用 Agent 可能不再输出 `JSON` 或 `Text`，而是直接输出 `Python Code` 来操作一切（Excel, Browser, OS）。

---

### 总结

Voyager 告诉我们：**真正的智能体不仅仅是解决问题的工具，更是提出问题并自我进化的实体。**
它通过 **Curriculum (提问)** 和 **Coding (解答)** 的循环，在没有任何人类监督的情况下，实现了知识的自举（Bootstrapping）。

下一课，我们将进入 **Agent 微调 (AgentTuning)**，探讨如何把像 Voyager 这样的大模型 Agent 的能力，蒸馏到轻量级模型中，实现低成本部署。
