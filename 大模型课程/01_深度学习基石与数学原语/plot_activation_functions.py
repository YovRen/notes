"""
Activation Functions Comparison: Φ(x), GeLU, ReLU, Sigmoid, Swish, Tanh
Including function curves and derivative curves
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import norm

plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# Define x range
x = np.linspace(-4, 4, 1000)

# ============================================================
# Define activation functions
# ============================================================

# 1. Φ(x) - Standard Normal CDF
def phi(x):
    return norm.cdf(x)

# 2. GeLU
def gelu(x):
    return x * phi(x)

def gelu_derivative(x):
    return phi(x) + x * norm.pdf(x)

# 3. ReLU
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# 4. Leaky ReLU
def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

# 5. Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# 6. Swish / SiLU
def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

# 7. Tanh
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# ============================================================
# Plotting
# ============================================================

fig = plt.figure(figsize=(18, 12))

# ==================== Plot 1: Φ(x) vs Sigmoid ====================
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, phi(x), 'b-', linewidth=2.5, label=r'$\Phi(x)$ (Normal CDF)')
ax1.plot(x, norm.pdf(x), 'b--', linewidth=2, label=r'$\phi(x)$ (Normal PDF)')
ax1.plot(x, sigmoid(x), 'r-', linewidth=2.5, label=r'$\sigma(x)$ (Sigmoid)')
ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-0.1, 1.1)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title(r'$\Phi(x)$ vs Sigmoid $\sigma(x)$' + '\n(Very Similar!)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)

# ==================== Plot 2: All Activation Functions ====================
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, relu(x), 'r-', linewidth=2.5, label='ReLU')
ax2.plot(x, leaky_relu(x), 'r--', linewidth=2, label=r'Leaky ReLU ($\alpha$=0.1)')
ax2.plot(x, gelu(x), 'b-', linewidth=2.5, label=r'GeLU = $x \cdot \Phi(x)$')
ax2.plot(x, swish(x), 'g-', linewidth=2.5, label=r'Swish/SiLU = $x \cdot \sigma(x)$')
ax2.plot(x, tanh(x), 'm-', linewidth=2, label='Tanh')

ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-2, 4)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.set_title('Activation Functions Comparison', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=11)
ax2.grid(True, alpha=0.3)

# ==================== Plot 3: Negative Region Zoom ====================
ax3 = fig.add_subplot(2, 2, 3)
x_zoom = np.linspace(-3, 1.5, 500)

ax3.plot(x_zoom, relu(x_zoom), 'r-', linewidth=2.5, label='ReLU', alpha=0.8)
ax3.plot(x_zoom, gelu(x_zoom), 'b-', linewidth=2.5, label='GeLU')
ax3.plot(x_zoom, swish(x_zoom), 'g-', linewidth=2.5, label='Swish/SiLU')
ax3.plot(x_zoom, leaky_relu(x_zoom), 'r--', linewidth=2, label='Leaky ReLU', alpha=0.7)

# Highlight GeLU negative region
ax3.fill_between(x_zoom, relu(x_zoom), gelu(x_zoom), 
                  where=(x_zoom < 0), alpha=0.2, color='blue',
                  label='GeLU negative region (non-zero!)')
ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

# Mark minimum points
gelu_min_x = -0.75
swish_min_x = -1.28
ax3.scatter([gelu_min_x], [gelu(gelu_min_x)], color='blue', s=80, zorder=5)
ax3.scatter([swish_min_x], [swish(swish_min_x)], color='green', s=80, zorder=5)
ax3.annotate(f'GeLU min\n({gelu_min_x:.2f}, {gelu(gelu_min_x):.2f})', 
             xy=(gelu_min_x, gelu(gelu_min_x)), xytext=(-2.5, -0.5),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'))

ax3.set_xlim(-3, 1.5)
ax3.set_ylim(-0.5, 1.5)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('f(x)', fontsize=12)
ax3.set_title('Negative Region Zoom\nGeLU/Swish allow small negatives (avoid Dead Neuron)', fontsize=14, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)

# ==================== Plot 4: Derivatives Comparison ====================
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, relu_derivative(x), 'r-', linewidth=2.5, label="ReLU' (0 or 1)")
ax4.plot(x, gelu_derivative(x), 'b-', linewidth=2.5, label="GeLU' (smooth)")
ax4.plot(x, swish_derivative(x), 'g-', linewidth=2.5, label="Swish' (smooth)")
ax4.plot(x, sigmoid_derivative(x), 'orange', linewidth=2, label="Sigmoid' (max=0.25)")
ax4.plot(x, tanh_derivative(x), 'm-', linewidth=2, label="Tanh' (max=1)")

ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='y=1 (ideal gradient)')
ax4.axhline(y=0.25, color='orange', linestyle=':', alpha=0.5)
ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax4.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

ax4.set_xlim(-4, 4)
ax4.set_ylim(-0.2, 1.3)
ax4.set_xlabel('x', fontsize=12)
ax4.set_ylabel("f'(x)", fontsize=12)
ax4.set_title("Derivatives Comparison\n(Determines Vanishing Gradient)", fontsize=14, fontweight='bold')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)

# Add annotations
ax4.annotate('Sigmoid max derivative = 0.25\n-> Severe vanishing gradient!', 
             xy=(0, 0.25), xytext=(1.5, 0.4),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='orange'))
ax4.annotate("ReLU' = 1 for x > 0\n-> No gradient decay!", 
             xy=(2, 1), xytext=(2.5, 0.7),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('activation_functions_comparison.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("Image saved: activation_functions_comparison.png")

# ============================================================
# Print key values
# ============================================================
print("\n" + "=" * 50)
print("Key Values Comparison")
print("=" * 50)

print("\nFunction values at x = -1:")
print(f"  ReLU(-1)  = {relu(-1):.4f}      <- Completely blocked")
print(f"  GeLU(-1)  = {gelu(-1):.4f}   <- Allows small negative!")
print(f"  Swish(-1) = {swish(-1):.4f}   <- Allows small negative!")

print("\nDerivatives at x = 0:")
print(f"  Sigmoid'(0) = {sigmoid_derivative(0):.4f}  <- Very small! Vanishing gradient!")
print(f"  Tanh'(0)    = {tanh_derivative(0):.4f}")
print(f"  GeLU'(0)    = {gelu_derivative(0):.4f}")
print(f"  Swish'(0)   = {swish_derivative(0):.4f}")
print(f"  ReLU'(0)    = undefined (0 from left, 1 from right)")

print("\n" + "=" * 50)
print("Summary:")
print("=" * 50)
print("""
| Function | Derivative Range | Vanishing Gradient Risk |
|----------|-----------------|------------------------|
| Sigmoid  | (0, 0.25]       | HIGH - max only 0.25   |
| Tanh     | (0, 1]          | MEDIUM - saturates     |
| ReLU     | {0, 1}          | LOW - but Dead Neuron  |
| GeLU     | smooth, ~1      | LOW - smooth gradient  |
| Swish    | smooth, ~1      | LOW - smooth gradient  |
""")
