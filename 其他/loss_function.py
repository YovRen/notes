import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf  # 用于GELU的精确计算

# --- 1. 定义激活函数 ---


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))


def gelu(x):
    # 使用scipy的erf函数进行精确计算
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    # 近似版本，如果不想引入scipy:
    # return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def swish(x):
    return x * sigmoid(x)

# --- 2. 定义损失函数 ---


def mse(y_true, y_pred):
    return (y_true - y_pred)**2


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)


def binary_cross_entropy(y_true, y_pred, epsilon=1e-10):
    # 裁剪y_pred以避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# --- 3. 准备绘图数据 ---
# 激活函数输入范围
x_activations = np.linspace(-5, 5, 400)

# 损失函数输入范围
# 对于MSE/MAE，我们通常看误差 e = y_pred - y_true
error_values = np.linspace(-3, 3, 400)
# 对于交叉熵，我们看预测概率 y_pred (0到1之间)
y_pred_probs = np.linspace(0.01, 0.99, 400)  # 避免log(0)或log(1)

# --- 4. 绘图 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- 绘制激活函数 ---
ax1 = axes[0]
ax1.plot(x_activations, sigmoid(x_activations), label='Sigmoid', lw=2)
ax1.plot(x_activations, tanh(x_activations), label='Tanh', lw=2)
ax1.plot(x_activations, relu(x_activations), label='ReLU', lw=2)
ax1.plot(x_activations, leaky_relu(x_activations), label='Leaky ReLU (α=0.01)', lw=2, linestyle='--')
ax1.plot(x_activations, elu(x_activations), label='ELU (α=1)', lw=2)
ax1.plot(x_activations, gelu(x_activations), label='GELU', lw=2, linestyle=':')
ax1.plot(x_activations, swish(x_activations), label='Swish', lw=2)

ax1.set_title('Activation Functions')
ax1.set_xlabel('Input (x)')
ax1.set_ylabel('Output')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)
ax1.legend()
ax1.set_ylim(-2, 2)  # 限制Y轴范围以便更好地对比

# --- 绘制损失函数 ---
ax2 = axes[1]

# MSE/MAE (y_true - y_pred = error)
ax2.plot(error_values, mse(0, error_values), label='MSE (Error^2)', lw=2)
ax2.plot(error_values, mae(0, error_values), label='MAE (|Error|)', lw=2)

# Binary Cross-Entropy (as a function of y_pred for specific y_true)
# When y_true = 1, loss = -log(y_pred)
ax2.plot(y_pred_probs, binary_cross_entropy(1, y_pred_probs), label='BCE (y_true=1)', lw=2, linestyle='--')
# When y_true = 0, loss = -log(1 - y_pred)
ax2.plot(y_pred_probs, binary_cross_entropy(0, y_pred_probs), label='BCE (y_true=0)', lw=2, linestyle=':')


ax2.set_title('Loss Functions')
ax2.set_xlabel('Prediction/Error')
ax2.set_ylabel('Loss Value')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axhline(0, color='black', linewidth=0.5)
# 限制Y轴以避免交叉熵在接近0或1时无限大
ax2.set_ylim(0, 4)
ax2.legend()

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()
