"""
可视化 PyTorch 的计算图 (DAG)
"""
import torch
import torch.nn as nn
from torchviz import make_dot

# ============================================================
# 例子1：简单的线性计算 L = (Wx + b)²
# ============================================================
print("=" * 50)
print("例子1：L = sum((Wx + b)²)")
print("=" * 50)

x = torch.randn(3, requires_grad=True)
W = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

h = W @ x + b      # 线性变换
y = h ** 2         # 平方
L = y.sum()        # 求和得到标量损失

# 生成 DAG 图
dot = make_dot(L, params={"x": x, "W": W, "b": b, "h": h, "y": y, "L": L})
dot.render("dag_simple", format="png", cleanup=True)
print("已保存: dag_simple.png")

# ============================================================
# 例子2：两层神经网络
# ============================================================
print("\n" + "=" * 50)
print("例子2：两层神经网络")
print("=" * 50)

x = torch.randn(4, requires_grad=True)
W1 = torch.randn(8, 4, requires_grad=True)
W2 = torch.randn(2, 8, requires_grad=True)

h1 = torch.relu(W1 @ x)    # 第一层 + ReLU
h2 = W2 @ h1               # 第二层
L = h2.sum()               # 损失

dot = make_dot(L, params={"x": x, "W1": W1, "W2": W2})
dot.render("dag_2layer", format="png", cleanup=True)
print("已保存: dag_2layer.png")

# ============================================================
# 例子3：简化的 Attention (最重要！)
# ============================================================
print("\n" + "=" * 50)
print("例子3：简化的 Self-Attention")
print("=" * 50)

# 简化版：单头 Attention
seq_len, d_model = 4, 8

x = torch.randn(seq_len, d_model, requires_grad=True)  # 输入序列
Wq = torch.randn(d_model, d_model, requires_grad=True)
Wk = torch.randn(d_model, d_model, requires_grad=True)
Wv = torch.randn(d_model, d_model, requires_grad=True)

Q = x @ Wq  # Query
K = x @ Wk  # Key
V = x @ Wv  # Value

# Attention 计算
scores = Q @ K.T / (d_model ** 0.5)  # 缩放点积
attn_weights = torch.softmax(scores, dim=-1)  # Softmax
output = attn_weights @ V  # 加权求和

L = output.sum()

dot = make_dot(L, params={"x": x, "Wq": Wq, "Wk": Wk, "Wv": Wv, 
                          "Q": Q, "K": K, "V": V, "scores": scores,
                          "attn": attn_weights, "out": output})
dot.render("dag_attention", format="png", cleanup=True)
print("已保存: dag_attention.png")

# ============================================================
# 例子4：打印节点信息（不用图形化）
# ============================================================
print("\n" + "=" * 50)
print("例子4：手动遍历计算图")
print("=" * 50)

def print_graph(tensor, indent=0):
    """递归打印计算图结构"""
    prefix = "  " * indent
    if tensor.grad_fn is None:
        print(f"{prefix}└─ [叶子节点] shape={list(tensor.shape)}")
        return
    
    print(f"{prefix}└─ {tensor.grad_fn.__class__.__name__}")
    for child, _ in tensor.grad_fn.next_functions:
        if child is not None:
            # 创建一个临时tensor来访问
            print_graph_fn(child, indent + 1)

def print_graph_fn(grad_fn, indent=0):
    """递归打印 grad_fn"""
    prefix = "  " * indent
    print(f"{prefix}└─ {grad_fn.__class__.__name__}")
    
    for child, _ in grad_fn.next_functions:
        if child is not None:
            print_graph_fn(child, indent + 1)

# 简单例子
print("\n计算图结构 (L = sum((Wx + b)²)):\n")
x = torch.randn(3, requires_grad=True)
W = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
L = ((W @ x + b) ** 2).sum()

print("L")
print_graph_fn(L.grad_fn, 0)

print("\n✅ 完成！请查看生成的 PNG 图片。")
