"""
用纯文本和 matplotlib 可视化 PyTorch 计算图 (DAG)
不需要安装 graphviz
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 手动遍历并打印 PyTorch 计算图
# ============================================================
print("=" * 60)
print("  PyTorch 计算图 (DAG) 结构可视化")
print("=" * 60)

def get_graph_structure(tensor, graph_dict=None, parent=None):
    """递归获取计算图结构"""
    if graph_dict is None:
        graph_dict = {"nodes": [], "edges": []}
    
    if tensor.grad_fn is None:
        node_name = f"Leaf_{id(tensor)}"
        if node_name not in [n[0] for n in graph_dict["nodes"]]:
            graph_dict["nodes"].append((node_name, "leaf", list(tensor.shape)))
        if parent:
            graph_dict["edges"].append((node_name, parent))
        return graph_dict
    
    node_name = f"{tensor.grad_fn.__class__.__name__}_{id(tensor.grad_fn)}"
    if node_name not in [n[0] for n in graph_dict["nodes"]]:
        graph_dict["nodes"].append((node_name, "op", tensor.grad_fn.__class__.__name__))
    
    if parent:
        graph_dict["edges"].append((node_name, parent))
    
    for child, _ in tensor.grad_fn.next_functions:
        if child is not None:
            get_graph_structure_fn(child, graph_dict, node_name)
    
    return graph_dict

def get_graph_structure_fn(grad_fn, graph_dict, parent):
    """递归处理 grad_fn"""
    node_name = f"{grad_fn.__class__.__name__}_{id(grad_fn)}"
    
    if node_name not in [n[0] for n in graph_dict["nodes"]]:
        graph_dict["nodes"].append((node_name, "op", grad_fn.__class__.__name__))
    
    graph_dict["edges"].append((node_name, parent))
    
    for child, _ in grad_fn.next_functions:
        if child is not None:
            get_graph_structure_fn(child, graph_dict, node_name)

def print_dag_text(tensor, name="L"):
    """以文本形式打印 DAG"""
    print(f"\n📊 计算图 (从 {name} 反向追溯):")
    print("-" * 50)
    
    def _print(grad_fn, indent=0, visited=None):
        if visited is None:
            visited = set()
        
        prefix = "│  " * indent
        
        if grad_fn is None:
            return
        
        fn_id = id(grad_fn)
        fn_name = grad_fn.__class__.__name__
        
        if fn_id in visited:
            print(f"{prefix}├─ {fn_name} (已访问)")
            return
        visited.add(fn_id)
        
        print(f"{prefix}├─ {fn_name}")
        
        for i, (child, _) in enumerate(grad_fn.next_functions):
            if child is not None:
                _print(child, indent + 1, visited)
            else:
                print(f"{prefix}│  └─ [叶子节点/输入]")
    
    _print(tensor.grad_fn)
    print("-" * 50)

# ============================================================
# 例子1: 简单计算 L = sum((Wx + b)²)
# ============================================================
print("\n" + "=" * 60)
print("  例子1: L = sum((Wx + b)²)")
print("=" * 60)

x = torch.randn(3, requires_grad=True)
W = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

h = W @ x       # 矩阵乘法
z = h + b       # 加偏置
y = z ** 2      # 平方
L = y.sum()     # 求和

print_dag_text(L, "L")

print("""
对应的 DAG 图示 (前向 → / 反向 ←):

    ┌───────────────────────────────────────────────────────────┐
    │                    前向传播方向 →                          │
    │                                                           │
    │    x [3]          W [2,3]         b [2]                   │
    │      │              │               │                     │
    │      └──────┬───────┘               │                     │
    │             ▼                       │                     │
    │         ┌───────┐                   │                     │
    │         │ MatMul │ ← MmBackward     │                     │
    │         └───┬───┘                   │                     │
    │             │ h [2]                 │                     │
    │             └─────────┬─────────────┘                     │
    │                       ▼                                   │
    │                  ┌─────────┐                              │
    │                  │   Add   │ ← AddBackward                │
    │                  └────┬────┘                              │
    │                       │ z [2]                             │
    │                       ▼                                   │
    │                  ┌─────────┐                              │
    │                  │  Pow(2) │ ← PowBackward                │
    │                  └────┬────┘                              │
    │                       │ y [2]                             │
    │                       ▼                                   │
    │                  ┌─────────┐                              │
    │                  │   Sum   │ ← SumBackward                │
    │                  └────┬────┘                              │
    │                       │                                   │
    │                       ▼                                   │
    │                    L (标量)                                │
    │                                                           │
    │                    ← 反向传播方向                          │
    └───────────────────────────────────────────────────────────┘
""")

# ============================================================
# 例子2: 两层神经网络
# ============================================================
print("\n" + "=" * 60)
print("  例子2: 两层神经网络 + ReLU")
print("=" * 60)

x = torch.randn(4, requires_grad=True)
W1 = torch.randn(8, 4, requires_grad=True)
W2 = torch.randn(2, 8, requires_grad=True)

h1 = torch.relu(W1 @ x)    # 第一层 + ReLU
h2 = W2 @ h1               # 第二层
L = h2.sum()

print_dag_text(L, "L")

print("""
DAG 图示:

    x [4]           W1 [8,4]                    W2 [2,8]
      │               │                           │
      └───────┬───────┘                           │
              ▼                                   │
         ┌─────────┐                              │
         │ MatMul  │ (W1 @ x)                     │
         └────┬────┘                              │
              │ [8]                               │
              ▼                                   │
         ┌─────────┐                              │
         │  ReLU   │ max(0, x)                    │
         └────┬────┘                              │
              │ h1 [8]                            │
              └────────────────┬──────────────────┘
                               ▼
                          ┌─────────┐
                          │ MatMul  │ (W2 @ h1)
                          └────┬────┘
                               │ h2 [2]
                               ▼
                          ┌─────────┐
                          │   Sum   │
                          └────┬────┘
                               │
                               ▼
                            L (标量)
""")

# ============================================================
# 例子3: Self-Attention (最重要!)
# ============================================================
print("\n" + "=" * 60)
print("  例子3: Self-Attention 的 DAG")
print("=" * 60)

seq_len, d = 4, 8

x = torch.randn(seq_len, d, requires_grad=True)
Wq = torch.randn(d, d, requires_grad=True)
Wk = torch.randn(d, d, requires_grad=True)
Wv = torch.randn(d, d, requires_grad=True)

Q = x @ Wq
K = x @ Wk
V = x @ Wv

scores = Q @ K.T / (d ** 0.5)
attn = torch.softmax(scores, dim=-1)
out = attn @ V
L = out.sum()

print_dag_text(L, "L")

print("""
Self-Attention 的 DAG 图示:

                            x [seq, d]
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
                ┌──────┐   ┌──────┐    ┌──────┐
          Wq ──▶│MatMul│   │MatMul│◀── Wk    │MatMul│◀── Wv
                └──┬───┘   └──┬───┘    └──┬───┘
                   │          │           │
                   ▼          ▼           │
                Q [seq,d]  K [seq,d]      │
                   │          │           │
                   │    ┌─────┘           │
                   │    │ K.T             │
                   ▼    ▼                 │
               ┌──────────┐               │
               │ MatMul   │ Q @ K.T       │
               └────┬─────┘               │
                    │ [seq, seq]          │
                    ▼                     │
               ┌──────────┐               │
               │  ÷ √d    │ 缩放          │
               └────┬─────┘               │
                    │                     │
                    ▼                     │
               ┌──────────┐               │
               │ Softmax  │ 注意力权重    │
               └────┬─────┘               │
                    │ attn [seq, seq]     │
                    │                     │
                    └─────────┬───────────┘
                              ▼  V [seq, d]
                         ┌──────────┐
                         │ MatMul   │ attn @ V
                         └────┬─────┘
                              │ output [seq, d]
                              ▼
                         ┌──────────┐
                         │   Sum    │
                         └────┬─────┘
                              │
                              ▼
                           L (标量)

⚠️ 注意：这个 DAG 的复杂度：
   - scores 矩阵是 [seq × seq]，当 seq=4096 时占用巨大显存
   - FlashAttention 的核心就是避免显式存储这个矩阵
""")

# ============================================================
# 用 Matplotlib 画一个简单的 DAG
# ============================================================
print("\n正在生成 DAG 可视化图片...")

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 11)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('计算图 (DAG) 示例: L = sum((Wx + b)²)', fontsize=16, fontweight='bold')

# 定义节点位置
nodes = {
    'x': (2, 10, '[3]', 'lightblue'),
    'W': (5, 10, '[2,3]', 'lightblue'),
    'b': (8, 10, '[2]', 'lightblue'),
    'MatMul': (3.5, 8, 'W @ x', 'lightyellow'),
    'Add': (5.5, 6, '+ b', 'lightyellow'),
    'Pow': (5.5, 4, 'x²', 'lightyellow'),
    'Sum': (5.5, 2, 'sum()', 'lightyellow'),
    'L': (5.5, 0, 'Loss', 'lightcoral'),
}

# 画节点
for name, (x, y, label, color) in nodes.items():
    box = FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, 
                         boxstyle="round,pad=0.05,rounding_size=0.2",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, f'{name}\n{label}', ha='center', va='center', fontsize=10, fontweight='bold')

# 画边 (前向)
edges = [
    ('x', 'MatMul'),
    ('W', 'MatMul'),
    ('MatMul', 'Add'),
    ('b', 'Add'),
    ('Add', 'Pow'),
    ('Pow', 'Sum'),
    ('Sum', 'L'),
]

for start, end in edges:
    x1, y1 = nodes[start][0], nodes[start][1] - 0.4
    x2, y2 = nodes[end][0], nodes[end][1] + 0.4
    
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

# 添加图例
ax.text(0, 1, '前向传播 →', fontsize=12, color='blue', fontweight='bold')
ax.text(0, 0, '反向传播 ←', fontsize=12, color='red', fontweight='bold')

# 画反向传播的箭头（虚线）
for start, end in reversed(edges):
    x1, y1 = nodes[end][0] + 0.3, nodes[end][1] + 0.4
    x2, y2 = nodes[start][0] + 0.3, nodes[start][1] - 0.4
    
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, linestyle='--'))

plt.tight_layout()
plt.savefig('dag_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\n✅ 图片已保存: dag_visualization.png")
