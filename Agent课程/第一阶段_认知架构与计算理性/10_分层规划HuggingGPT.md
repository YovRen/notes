# 第10课：分层规划 (HuggingGPT)

**关键词**：Hierarchy, Task Decomposition, DAG依赖图, 拓扑排序, Controller/Worker

---

## 笔记区域

你好。这是《AI Agent 深度架构与数学原理》的第十课。

在上节课（LLM+P）中，我们将规划外包给了符号求解器（PDDL Planner），这解决了逻辑严密性问题，但仅限于封闭域（如积木世界）。
现实世界是**多模态（Multi-modal）**且**开放域（Open Domain）**的。用户可能会问：“请描述这张图片，并根据描述生成一段配乐。” 这涉及视觉理解、文本生成、音频合成。

**背景驱动**：

* **挑战 (Challenge)**：单体 LLM（如 GPT-4）虽然是全能选手，但在特定领域的性能（如生成高质量图像、处理特定格式视频）往往不如专用模型（如 Stable Diffusion, Whisper）。且 LLM 无法直接处理非文本输入/输出流。
* **突破点 (Breakthrough)**：**HuggingGPT (Shen et al., NeurIPS 2023)** 提出的**分层规划（Hierarchical Planning）**与**中控架构（Controller Architecture）**。
* **核心思想**：LLM 不再是“干活的人”，而是“包工头（Controller/Scheduler）”。它利用 Hugging Face 上成千上万的专家模型（Expert Models）作为工具，通过规划将复杂任务拆解为 DAG（有向无环图），调度专家模型协同工作。

---

# 🧠 第10课：分层规划 (HuggingGPT)

### 1. 理论核心：任务分解与 DAG 调度

#### 1.1 数学定义：任务依赖图

我们将一个复杂的用户请求 $U$ 建模为一个**有向无环图 (DAG)**，记为 $\mathcal{G} = \langle \mathcal{T}, \mathcal{E} \rangle$。

1. **任务集合 ($\mathcal{T}$)**：
   $U$ 被 LLM 分解为一系列子任务 $\{t_1, t_2, \dots, t_n\}$。
   每个子任务 $t_i$ 是一个元组 $\langle \text{task\_type}, \text{args}, \text{dep} \rangle$。

   * $\text{task\_type}$: 如 `image-to-text`, `text-to-speech`。
   * $\text{dep}$: 依赖列表，即哪些任务的输出是当前任务的输入。
2. **依赖边 ($\mathcal{E}$)**：
   如果 $t_i$ 的输出是 $t_j$ 的输入，则存在边 $(t_i, t_j) \in \mathcal{E}$。
   这决定了拓扑排序（Topological Sort）和并行执行的可能性。

#### 1.2 模型选择概率

对于每个子任务 $t_i$，我们需要从模型库 $\mathcal{M}$ 中选择最合适的模型 $m_{ij}$。
这不仅是一个检索问题，还是一个推理问题。我们计算选择概率：

$$
P(m | t_i, \mathcal{C}) \propto \text{Sim}(E(t_i), E(Desc(m))) \cdot P_{LLM}(\text{Select } m | t_i, \mathcal{C})
$$

* $\text{Sim}(\cdot)$: 任务描述与模型描述的语义相似度（Embedding Similarity）。
* $P_{LLM}$: LLM 根据上下文 $\mathcal{C}$（如模型下载量、性能指标）进行的二次排序。

#### 1.3 资源分配与结果聚合

执行过程是一个函数复合：

$$
R = \text{Aggregator}(\{ \text{Exec}(m_{k}, \text{Args}_k) \mid \forall k \in \text{TopologicalSort}(\mathcal{G}) \})
$$

其中 $\text{Exec}$ 涉及不同模态数据的张量流转（Tensor Flow）。

---

### 2. 架构解剖与工程应用

#### 2.1 四阶段流水线 (The 4-Stage Pipeline)

HuggingGPT 的架构极其经典，被后续无数 Multi-modal Agent 效仿：

1. **Task Planning (任务规划)**: LLM 解析 Prompt，生成结构化的 Task List。
2. **Model Selection (模型选择)**: 根据 Task Type，结合 RAG 从 Model Hub 中检索 Top-K 模型，由 LLM 最终拍板。
3. **Task Execution (任务执行)**: 动态调用本地或云端的推理端点（Inference Endpoints），处理参数依赖（Resource Dependency）。
4. **Response Generation (响应生成)**: 收集所有执行结果（图片路径、音频文件、文本），由 LLM 汇总并向用户汇报。

#### 2.2 系统架构图 (Mermaid)

```mermaid
graph TD
    User[User Request] --> Controller[LLM Controller]
  
    subgraph Stage 1: Planning
    Controller -->|Decompose| TaskQueue[Task List: [t1, t2, t3]]
    end
  
    subgraph Stage 2: Selection
    TaskQueue --> Selector{Model Selector}
    DB[(HuggingFace Hub<br>Model Descriptions)] -.-> Selector
    Selector -->|Assign| T1_M[t1: DETR (Object Det)]
    Selector -->|Assign| T2_M[t2: ViT-GPT2 (Caption)]
    end
  
    subgraph Stage 3: Execution
    T1_M -->|Output: Bounding Box| Context
    T2_M -->|Output: Text| Context
    Context -->|Dependency| T3_M[t3: Stable Diffusion]
    T3_M -->|Output: Image| Results
    end
  
    subgraph Stage 4: Response
    Results --> Summarizer[LLM Summarizer]
    Summarizer --> Final[Final Response]
    end
```

#### 2.3 工程应用：输入输出详解

**场景**：用户上传一张图 `a.jpg`，说：“请数一下图里有几个人，然后根据这个数量写一首诗，最后读出来。”

1. **LLM Input**:
   * System Prompt: 定义了 Task parsing 的 JSON 格式。
   * User Prompt: "Image: /tmp/a.jpg. Count people, write poem based on count, generate audio."
2. **LLM Output (The Plan)**:

   ```json
   [
     {"id": 0, "task": "object-detection", "args": ["/tmp/a.jpg"], "dep": [-1]},
     {"id": 1, "task": "visual-question-answering", "args": ["/tmp/a.jpg", "How many people?"], "dep": [-1]},
     {"id": 2, "task": "text-generation", "args": ["Write a poem about {1_output} people"], "dep": [1]},
     {"id": 3, "task": "text-to-speech", "args": ["{2_output}"], "dep": [2]}
   ]
   ```

   *(注：HuggingGPT 实际上会解析依赖关系，如 `{id}_output`)*
3. **后续操作**:
   * **Parsing**: 提取 JSON。
   * **Dependency Resolution**: 发现 Task 2 依赖 Task 1 的结果。Task 0 和 Task 1 可以并行（如果没有资源冲突）。
   * **Execution**:
     * 运行 Task 1 (VQA) -> 得到 "3"。
     * 替换 Task 2 参数 -> "Write a poem about 3 people"。
     * 运行 Task 2 (LLM/GPT2) -> 得到诗歌文本。
     * 运行 Task 3 (TTS) -> 生成 `.wav` 文件。

---

### 3. Code & Engineering：实现 DAG 任务调度器

为了让研三学生理解**分层规划**的核心，我们实现一个简化的 **Dependency Aware Scheduler**。

```python
import json
import time
from typing import List, Dict, Any

class TaskNode:
    def __init__(self, id: int, task_type: str, args: List[Any], dependencies: List[int]):
        self.id = id
        self.task_type = task_type
        self.args = args
        self.dependencies = dependencies # List of parent Task IDs
        self.status = "pending" # pending, running, completed
        self.output = None

class HierarchicalPlanner:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.tasks: Dict[int, TaskNode] = {}

    def parse_plan(self, plan_json: str):
        """ 将 LLM 生成的 JSON 解析为 Task Graph """
        plan_list = json.loads(plan_json)
        for item in plan_list:
            node = TaskNode(
                id=item['id'],
                task_type=item['task'],
                args=item['args'],
                dependencies=item['dep']
            )
            self.tasks[node.id] = node

    def execute_task(self, task: TaskNode):
        """ 模拟执行 Expert Model """
        print(f"🚀 Executing Task {task.id}: {task.task_type} with args {task.args}")
        # 这里是实际调用 HF API 或 本地模型的地方
        time.sleep(1) # Simulate latency
        return f"Result_of_{task.task_type}"

    def resolve_arguments(self, task: TaskNode):
        """ 核心逻辑：参数依赖注入 """
        # 将参数中的占位符 <id>_output 替换为实际结果
        new_args = []
        for arg in task.args:
            if isinstance(arg, str) and "_output" in arg:
                # 简单的解析逻辑，实际需正则
                dep_id = int(arg.split("_")[0].replace("<", "").replace(">", ""))
                if dep_id in self.tasks and self.tasks[dep_id].output:
                    actual_val = self.tasks[dep_id].output
                    arg = arg.replace(f"<{dep_id}>_output", str(actual_val))
            new_args.append(arg)
        task.args = new_args

    def run_dag(self):
        """ 拓扑排序执行 """
        completed_count = 0
        total_tasks = len(self.tasks)
      
        while completed_count < total_tasks:
            # 寻找所有依赖已满足且未执行的任务 (Ready Tasks)
            ready_tasks = []
            for t_id, task in self.tasks.items():
                if task.status == "pending":
                    deps_met = all(self.tasks[d_id].status == "completed" for d_id in task.dependencies if d_id != -1)
                    if deps_met:
                        ready_tasks.append(task)
          
            if not ready_tasks:
                raise Exception("Deadlock detected or Cycle in graph!")

            # 并行执行 (这里简化为串行循环，但在工程中应用 ThreadPool)
            for task in ready_tasks:
                task.status = "running"
                self.resolve_arguments(task) # 注入上游结果
                task.output = self.execute_task(task)
                task.status = "completed"
                completed_count += 1
                print(f"✅ Task {task.id} Finished. Output: {task.output}")

# --- Simulation ---
# 假设 LLM 生成了如下 Plan
plan_str = """
[
    {"id": 0, "task": "object_detection", "args": ["image.jpg"], "dep": [-1]},
    {"id": 1, "task": "tts", "args": ["I found <0>_output in the image"], "dep": [0]}
]
"""

# planner = HierarchicalPlanner(mock_llm)
# planner.parse_plan(plan_str)
# planner.run_dag()
```

---

### 4. Paper Driven：核心论文与贡献

1. **Shen et al. (NeurIPS 2023)**: *HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face*.
   * **核心贡献**：提出了 **LLM-as-a-Controller** 的范式。证明了 LLM 可以通过 API 描述来调度多模态模型。
   * **关键点**：解决了 Context Window 限制问题。通过 RAG 检索模型描述，LLM 不需要知道所有模型的 API，只需看到 Top-K 相关的模型描述即可。
2. **Lu et al. (NeurIPS 2023)**: *Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models*.
   * **对比**：HuggingGPT 侧重于通用任务调度；Chameleon 侧重于**科学推理**和**组合式推理**，它引入了更严格的模块清单（Inventory）和查询生成器。
3. **Schick et al. (2023)**: *Toolformer: Language Models Can Teach Themselves to Use Tools*.
   * **区别**：Toolformer 是通过**微调 (Fine-tuning)** 让模型学会调用 API；HuggingGPT 是通过**上下文学习 (In-Context Learning)** 做到的。HuggingGPT 更灵活，Toolformer 更快更准。

---

### 5. Critical Thinking：批判性分析

HuggingGPT 类的架构非常炫酷，但在工业界落地极难。

1. **Latency (延迟爆炸)**:

   * **瓶颈**：Pipeline 太长。Task Parsing (LLM) + Model Selection (LLM + Embedding) + Execution (Network/GPU) + Summary (LLM)。处理一个请求可能需要 30秒+。
   * **解决思路**：**Task Caching (任务缓存)**。对于相似的 Prompt，直接复用解析好的 Plan DAG，跳过前两步。
2. **Robustness (依赖断裂)**:

   * **瓶颈**：如果上游模型（如 Object Detection）输出格式变了（从 JSON 变成 XML），下游模型（TTS）直接报错。
   * **解决思路**：**Type Checking & Middleware**。在 DAG 节点之间增加数据适配层（Adapter），强制类型转换。
3. **Cost (成本)**:

   * **瓶颈**：调用多个专家模型和多次 GPT-4 的成本极高。
   * **解决思路**：**Distillation (蒸馏)**。将 "Planning + Selection" 的能力蒸馏给一个小模型（如 Llama-3-8B），作为专用的 Controller。

---

### 6. 前沿扩展

* **Multi-Agent Hierarchy**:
  * 将 Controller 升级为 **Boss Agent**，将每个 Task 升级为 **Worker Agent**。
  * Boss 负责分发任务，Worker 负责寻找具体的工具并执行。如果 Worker 遇到困难，可以向 Boss 报错，Boss 重新规划。
* **Auto-Finetuning**:
  * 记录 HuggingGPT 成功的调用链（Prompt -> Plan -> Result）。用这些数据微调 LLM，使其内化“什么任务该用什么模型”，从而在未来省略 Model Selection 步骤，直接生成 Plan。

---

### 总结

分层规划（Hierarchical Planning）解决了 LLM **“全能但不精通”** 的问题。
通过 **DAG 调度** 和 **工具链编排**，我们构建了一个**多模态的神经系统**：LLM 是大脑皮层（负责规划），Expert Models 是小脑和感官（负责执行），而 DAG 是连接它们的神经束。

下一课，我们将深入 **自动课程学习 (Voyager)**，探讨 Agent 如何在没有人类干预的情况下，通过探索环境自我进化，习得新技能。
