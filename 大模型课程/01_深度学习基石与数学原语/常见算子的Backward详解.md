# 常见算子的 Forward 与 Backward 详解

> 符号约定：$\bar{y} = \frac{\partial L}{\partial y}$ 表示上游传来的梯度，$\bar{x} = \frac{\partial L}{\partial x}$ 表示需要传回去的梯度

---

## 📌 一、基础元素级操作 (Element-wise)

### 1.1 加法 (Add)

**Forward**: $y = x_1 + x_2$

**Backward**:
$$
\bar{x}_1 = \bar{y}, \quad \bar{x}_2 = \bar{y}
$$

> 💡 梯度直接复制传递，这就是 **Skip Connection (残差连接)** 能缓解梯度消失的原因！

---

### 1.2 标量乘法 (Scale)

**Forward**: $y = c \cdot x$（$c$ 是常数）

**Backward**:
$$
\bar{x} = c \cdot \bar{y}
$$

---

### 1.3 元素乘法 (Hadamard Product)

**Forward**: $y = x_1 \odot x_2$（逐元素相乘）

**Backward**:
$$
\bar{x}_1 = \bar{y} \odot x_2, \quad \bar{x}_2 = \bar{y} \odot x_1
$$

> 💡 谁的梯度，就乘以另一个的值

---

### 1.4 除法 (Division)

**Forward**: $y = \frac{x_1}{x_2}$

**Backward**:
$$
\bar{x}_1 = \frac{\bar{y}}{x_2}, \quad \bar{x}_2 = -\frac{\bar{y} \cdot x_1}{x_2^2} = -\frac{\bar{y} \cdot y}{x_2}
$$

---

### 1.5 幂运算 (Power)

**Forward**: $y = x^n$

**Backward**:
$$
\bar{x} = n \cdot x^{n-1} \cdot \bar{y}
$$

---

### 1.6 指数 (Exp)

**Forward**: $y = e^x$

**Backward**:
$$
\bar{x} = y \cdot \bar{y} = e^x \cdot \bar{y}
$$

> 💡 这就是为什么 exp 容易导致梯度爆炸

---

### 1.7 对数 (Log)

**Forward**: $y = \ln(x)$

**Backward**:
$$
\bar{x} = \frac{\bar{y}}{x}
$$

> 💡 当 $x \to 0$ 时，梯度会爆炸，这就是为什么需要 `log(x + eps)`

---

## 📌 二、激活函数 (Activation Functions)

### 2.1 ReLU

**Forward**: $y = \max(0, x)$

**Backward**:
$$
\bar{x} = \begin{cases} \bar{y} & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases} = \bar{y} \cdot \mathbf{1}_{x>0}
$$

```python
# PyTorch 实现
def relu_backward(grad_output, x):
    return grad_output * (x > 0).float()
```

> 💡 x ≤ 0 时梯度完全为 0，这是 "Dead ReLU" 问题的根源

---

### 2.2 Leaky ReLU

**Forward**: $y = \max(\alpha x, x)$（通常 $\alpha = 0.01$）

**Backward**:
$$
\bar{x} = \begin{cases} \bar{y} & \text{if } x > 0 \\ \alpha \cdot \bar{y} & \text{if } x \leq 0 \end{cases}
$$

---

### 2.3 Sigmoid

**Forward**: $y = \sigma(x) = \frac{1}{1 + e^{-x}}$

**Backward**:
$$
\bar{x} = \bar{y} \cdot y \cdot (1 - y) = \bar{y} \cdot \sigma(x)(1 - \sigma(x))
$$

```python
# PyTorch 实现
def sigmoid_backward(grad_output, y):
    return grad_output * y * (1 - y)
```

> 💡 当 $y \to 0$ 或 $y \to 1$ 时，梯度接近 0，导致**梯度消失**

---

### 2.4 Tanh

**Forward**: $y = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

**Backward**:
$$
\bar{x} = \bar{y} \cdot (1 - y^2)
$$

```python
# PyTorch 实现
def tanh_backward(grad_output, y):
    return grad_output * (1 - y ** 2)
```

> 💡 和 Sigmoid 类似，在饱和区梯度接近 0

---

### 2.5 GeLU (GPT 系列使用)

**Forward**: $y = x \cdot \Phi(x)$，其中 $\Phi$ 是标准正态分布的 CDF

近似公式: $y \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$

**Backward** (精确形式):
$$
\bar{x} = \bar{y} \cdot \left[ \Phi(x) + x \cdot \phi(x) \right]
$$

其中 $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ 是标准正态 PDF

---

### 2.6 SiLU / Swish

**Forward**: $y = x \cdot \sigma(x)$

**Backward**:
$$
\bar{x} = \bar{y} \cdot \left[ \sigma(x) + x \cdot \sigma(x)(1-\sigma(x)) \right] = \bar{y} \cdot \left[ y + \sigma(x)(1-y) \right]
$$

---

### 2.7 SwiGLU (LLaMA 使用)

**Forward**: $y = \text{Swish}(xW_1) \odot (xW_2)$

**Backward**: 需要分别对 $W_1$, $W_2$ 和 $x$ 求梯度（复合运算）

---

## 📌 三、矩阵运算 (Matrix Operations)

### 3.1 矩阵乘法 (MatMul)

**Forward**: $Y = XW$，其中 $X \in \mathbb{R}^{m \times n}$, $W \in \mathbb{R}^{n \times p}$, $Y \in \mathbb{R}^{m \times p}$

**Backward**:
$$
\bar{X} = \bar{Y} W^T, \quad \bar{W} = X^T \bar{Y}
$$

```python
# PyTorch 实现
def matmul_backward(grad_output, X, W):
    grad_X = grad_output @ W.T   # [m, p] @ [p, n] = [m, n]
    grad_W = X.T @ grad_output   # [n, m] @ [m, p] = [n, p]
    return grad_X, grad_W
```

> 💡 这是深度学习中最核心的 backward！维度分析很重要

**维度检查口诀**：
- $\bar{X}$ 的 shape 必须和 $X$ 一样 → 用 $\bar{Y}W^T$
- $\bar{W}$ 的 shape 必须和 $W$ 一样 → 用 $X^T\bar{Y}$

---

### 3.2 带 Bias 的线性层

**Forward**: $Y = XW + b$

**Backward**:
$$
\bar{X} = \bar{Y} W^T, \quad \bar{W} = X^T \bar{Y}, \quad \bar{b} = \sum_{\text{batch}} \bar{Y}
$$

> 💡 Bias 的梯度是沿 batch 维度求和

---

### 3.3 转置 (Transpose)

**Forward**: $Y = X^T$

**Backward**:
$$
\bar{X} = \bar{Y}^T
$$

---

### 3.4 Reshape / View

**Forward**: $Y = \text{reshape}(X, \text{new\_shape})$

**Backward**:
$$
\bar{X} = \text{reshape}(\bar{Y}, \text{original\_shape})
$$

> 💡 Reshape 不改变数据，只改变形状，所以梯度形状也只是还原

---

## 📌 四、归一化层 (Normalization)

### 4.1 Layer Normalization

**Forward**: 
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta
$$

其中 $\mu = \frac{1}{n}\sum x_i$, $\sigma^2 = \frac{1}{n}\sum(x_i - \mu)^2$

**Backward** (较复杂):

$$
\bar{x}_i = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left( \bar{y}_i - \frac{1}{n}\sum_j \bar{y}_j - \frac{\hat{x}_i}{n}\sum_j \bar{y}_j \hat{x}_j \right)
$$

$$
\bar{\gamma} = \sum_i \bar{y}_i \hat{x}_i, \quad \bar{\beta} = \sum_i \bar{y}_i
$$

```python
# 简化的 PyTorch 实现
def layernorm_backward(grad_output, x, gamma, mean, var, eps=1e-5):
    N = x.shape[-1]
    std = (var + eps).sqrt()
    x_hat = (x - mean) / std
    
    # 对 gamma 和 beta 的梯度
    grad_gamma = (grad_output * x_hat).sum(dim=0)
    grad_beta = grad_output.sum(dim=0)
    
    # 对输入 x 的梯度 (复杂！)
    dx_hat = grad_output * gamma
    dvar = (dx_hat * (x - mean) * -0.5 * (var + eps)**(-1.5)).sum(dim=-1, keepdim=True)
    dmean = (dx_hat * -1/std).sum(dim=-1, keepdim=True) + dvar * (-2/N * (x - mean)).sum(dim=-1, keepdim=True)
    grad_x = dx_hat / std + dvar * 2/N * (x - mean) + dmean / N
    
    return grad_x, grad_gamma, grad_beta
```

---

### 4.2 RMSNorm (LLaMA 使用)

**Forward**:
$$
y_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}
$$

**Backward**:
$$
\bar{x}_i = \frac{\gamma}{\text{RMS}} \left( \bar{y}_i - \frac{x_i}{n \cdot \text{RMS}^2} \sum_j x_j \bar{y}_j \gamma \right)
$$

> 💡 RMSNorm 比 LayerNorm 简单：没有减均值，只有缩放

---

## 📌 五、Softmax 与损失函数

### 5.1 Softmax

**Forward**: $S_i = \frac{e^{x_i}}{\sum_j e^{x_j}}$

**Backward** (Jacobian 形式):
$$
\frac{\partial S_i}{\partial x_j} = S_i(\delta_{ij} - S_j)
$$

**VJP 形式** (实际计算):
$$
\bar{x}_j = S_j \left( \bar{S}_j - \sum_k \bar{S}_k S_k \right)
$$

```python
# PyTorch 实现
def softmax_backward(grad_output, softmax_output):
    # grad_output: 上游梯度 [batch, n]
    # softmax_output: forward 的输出 [batch, n]
    s = softmax_output
    # Σ(grad * s)
    sum_grad_s = (grad_output * s).sum(dim=-1, keepdim=True)
    # s * (grad - Σ(grad * s))
    grad_input = s * (grad_output - sum_grad_s)
    return grad_input
```

---

### 5.2 Cross Entropy Loss

**Forward** (带 Softmax):
$$
L = -\sum_i y_i \log(S_i)
$$

其中 $y$ 是 one-hot 标签，$S$ 是 softmax 输出

**Backward** (对 logits $x$):
$$
\bar{x}_i = S_i - y_i
$$

> 💡 这是一个**极其简洁**的结果！实际中 PyTorch 把 Softmax + CrossEntropy 融合成一个算子就是因为这个

```python
# PyTorch 实现
def cross_entropy_backward(softmax_output, target_one_hot):
    # 结果就是 softmax 输出减去真实标签！
    return softmax_output - target_one_hot
```

**例子**：
- 预测概率: $[0.7, 0.2, 0.1]$
- 真实标签: $[1, 0, 0]$（类别0）
- 梯度: $[0.7-1, 0.2-0, 0.1-0] = [-0.3, 0.2, 0.1]$

---

### 5.3 LogSoftmax + NLLLoss

**Forward**:
$$
\text{LogSoftmax}: \quad z_i = x_i - \log\sum_j e^{x_j}
$$
$$
\text{NLLLoss}: \quad L = -z_{\text{target}}
$$

**Backward** (对 logits $x$):
$$
\bar{x}_i = e^{z_i} - \mathbf{1}_{i=\text{target}} = S_i - y_i
$$

> 💡 和上面 CrossEntropy 结果一样，但数值更稳定

---

## 📌 六、Attention 相关

### 6.1 Scaled Dot-Product Attention

**Forward**:
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

设 $A = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})$，$O = AV$

**Backward**:

$$
\bar{V} = A^T \bar{O}
$$
$$
\bar{A} = \bar{O} V^T
$$
$$
\bar{(QK^T)} = \text{softmax\_backward}(\bar{A}) / \sqrt{d_k}
$$
$$
\bar{Q} = \bar{(QK^T)} K, \quad \bar{K} = \bar{(QK^T)}^T Q
$$

```python
# 简化实现
def attention_backward(grad_output, Q, K, V, attn_weights):
    d_k = Q.shape[-1]
    
    # dV = A^T @ grad_output
    grad_V = attn_weights.transpose(-2, -1) @ grad_output
    
    # dA = grad_output @ V^T
    grad_A = grad_output @ V.transpose(-2, -1)
    
    # 通过 softmax backward
    grad_scores = softmax_backward(grad_A, attn_weights) / math.sqrt(d_k)
    
    # dQ = grad_scores @ K
    grad_Q = grad_scores @ K
    
    # dK = grad_scores^T @ Q
    grad_K = grad_scores.transpose(-2, -1) @ Q
    
    return grad_Q, grad_K, grad_V
```

---

## 📌 七、其他常见操作

### 7.1 Dropout

**Forward**: $y = \frac{x \cdot m}{1-p}$，其中 $m \sim \text{Bernoulli}(1-p)$

**Backward**:
$$
\bar{x} = \frac{\bar{y} \cdot m}{1-p}
$$

> 💡 需要保存 mask $m$！同一个 mask 在 forward 和 backward 中使用

---

### 7.2 Embedding

**Forward**: $Y = \text{Embedding}[X]$（查表操作）

**Backward**: 
$$
\bar{\text{Embedding}}[i] = \sum_{j: X_j = i} \bar{Y}_j
$$

> 💡 梯度散射回原来的位置，用 `scatter_add` 实现

```python
def embedding_backward(grad_output, indices, num_embeddings, embedding_dim):
    grad_embedding = torch.zeros(num_embeddings, embedding_dim)
    grad_embedding.scatter_add_(0, indices.unsqueeze(-1).expand_as(grad_output), grad_output)
    return grad_embedding
```

---

### 7.3 Sum / Mean

**Forward (Sum)**: $y = \sum_i x_i$

**Backward**:
$$
\bar{x}_i = \bar{y}
$$

**Forward (Mean)**: $y = \frac{1}{n}\sum_i x_i$

**Backward**:
$$
\bar{x}_i = \frac{\bar{y}}{n}
$$

---

### 7.4 Max

**Forward**: $y = \max_i x_i$

**Backward**:
$$
\bar{x}_i = \begin{cases} \bar{y} & \text{if } x_i = y \\ 0 & \text{otherwise} \end{cases}
$$

> 💡 梯度只流向最大值位置，其他位置梯度为 0

---

### 7.5 Concatenate

**Forward**: $Y = [X_1; X_2; ...; X_n]$（沿某个维度拼接）

**Backward**:
$$
\bar{X}_i = \text{slice}(\bar{Y}, i)
$$

> 💡 把梯度切分回原来的形状

---

## 📌 八、快速参考表

| 算子 | Forward | Backward $\bar{x}$ |
|:---|:---|:---|
| Add | $y = x_1 + x_2$ | $\bar{x}_1 = \bar{y}, \bar{x}_2 = \bar{y}$ |
| Mul | $y = x_1 \cdot x_2$ | $\bar{x}_1 = \bar{y} \cdot x_2$ |
| MatMul | $Y = XW$ | $\bar{X} = \bar{Y}W^T, \bar{W} = X^T\bar{Y}$ |
| ReLU | $y = \max(0, x)$ | $\bar{x} = \bar{y} \cdot \mathbf{1}_{x>0}$ |
| Sigmoid | $y = \sigma(x)$ | $\bar{x} = \bar{y} \cdot y(1-y)$ |
| Tanh | $y = \tanh(x)$ | $\bar{x} = \bar{y} \cdot (1-y^2)$ |
| Softmax | $S_i = \frac{e^{x_i}}{\sum e^{x_j}}$ | $\bar{x}_j = S_j(\bar{S}_j - \sum_k \bar{S}_k S_k)$ |
| CrossEntropy | $L = -\sum y_i \log S_i$ | $\bar{x} = S - y$ |
| Exp | $y = e^x$ | $\bar{x} = y \cdot \bar{y}$ |
| Log | $y = \ln x$ | $\bar{x} = \bar{y} / x$ |
| Sum | $y = \sum x_i$ | $\bar{x}_i = \bar{y}$ |
| Mean | $y = \frac{1}{n}\sum x_i$ | $\bar{x}_i = \bar{y} / n$ |
| Max | $y = \max x_i$ | $\bar{x}_i = \bar{y}$ if $x_i = y$ else $0$ |

---

## 📌 九、验证代码

```python
import torch
import torch.nn.functional as F

def verify_gradients():
    """验证手动计算的梯度与 PyTorch 自动求导一致"""
    torch.manual_seed(42)
    
    # ===== 1. MatMul =====
    X = torch.randn(2, 3, requires_grad=True)
    W = torch.randn(3, 4, requires_grad=True)
    Y = X @ W
    loss = Y.sum()
    loss.backward()
    
    # 手动计算
    grad_Y = torch.ones_like(Y)
    grad_X_manual = grad_Y @ W.T
    grad_W_manual = X.T @ grad_Y
    
    print("=== MatMul ===")
    print(f"grad_X 差异: {(X.grad - grad_X_manual).abs().max():.2e}")
    print(f"grad_W 差异: {(W.grad - grad_W_manual).abs().max():.2e}")
    
    # ===== 2. ReLU =====
    x = torch.randn(5, requires_grad=True)
    y = F.relu(x)
    y.sum().backward()
    
    grad_x_manual = (x.detach() > 0).float()
    print("\n=== ReLU ===")
    print(f"grad_x 差异: {(x.grad - grad_x_manual).abs().max():.2e}")
    
    # ===== 3. Softmax + CrossEntropy =====
    x = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 3, 2])
    loss = F.cross_entropy(x, target)
    loss.backward()
    
    # 手动计算
    s = F.softmax(x.detach(), dim=-1)
    target_onehot = F.one_hot(target, 5).float()
    grad_x_manual = (s - target_onehot) / 3  # 除以 batch size
    
    print("\n=== Softmax + CrossEntropy ===")
    print(f"grad_x 差异: {(x.grad - grad_x_manual).abs().max():.2e}")

if __name__ == "__main__":
    verify_gradients()
```

---

## 🚀 总结

1. **核心原则**：$\bar{x} = \frac{\partial L}{\partial x} = \bar{y} \cdot \frac{\partial y}{\partial x}$

2. **维度口诀**：梯度的 shape 必须和原变量一样

3. **记忆技巧**：
   - 加法：梯度复制
   - 乘法：乘以对方
   - 矩阵乘：转置交换位置
   - 激活函数：乘以导数值

4. **工程意义**：理解 backward 才能写自定义 CUDA kernel（如 FlashAttention）
