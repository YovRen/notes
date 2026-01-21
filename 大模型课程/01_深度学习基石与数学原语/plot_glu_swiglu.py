"""
GLU vs SwiGLU Visualization
GLU and SwiGLU are NOT scalar functions, they are GATING mechanisms
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# Define functions
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def swish(x):
    return x * sigmoid(x)

def glu(gate, value):
    """GLU(gate, value) = sigmoid(gate) * value"""
    return sigmoid(gate) * value

def swiglu(gate, value):
    """SwiGLU(gate, value) = swish(gate) * value"""
    return swish(gate) * value

def geglu(gate, value):
    """GeGLU(gate, value) = gelu(gate) * value"""
    from scipy.stats import norm
    return gate * norm.cdf(gate) * value

# ============================================================
# Figure 1: 1D comparison (fix value, vary gate)
# ============================================================

fig = plt.figure(figsize=(18, 14))

# --- Plot 1: Gate functions ---
ax1 = fig.add_subplot(2, 3, 1)
x = np.linspace(-4, 4, 200)

ax1.plot(x, sigmoid(x), 'b-', linewidth=2.5, label=r'$\sigma(x)$ (GLU gate)')
ax1.plot(x, swish(x), 'g-', linewidth=2.5, label='Swish(x) (SwiGLU gate)')
ax1.plot(x, x * sigmoid(x) / (np.abs(x) + 0.1), 'r--', linewidth=2, label='Normalized Swish', alpha=0.7)

ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax1.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlabel('Gate Input', fontsize=12)
ax1.set_ylabel('Gate Output', fontsize=12)
ax1.set_title('Gate Functions Comparison\nSigmoid (GLU) vs Swish (SwiGLU)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Add annotation
ax1.annotate('Sigmoid: always in [0,1]\n-> "soft" gate', 
             xy=(2, sigmoid(2)), xytext=(0.5, 1.2),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='blue'))
ax1.annotate('Swish: can be > 1 or < 0\n-> more expressive', 
             xy=(3, swish(3)), xytext=(1, 0.3),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='green'))

# --- Plot 2: Output comparison (value=1) ---
ax2 = fig.add_subplot(2, 3, 2)
value = 1.0
ax2.plot(x, glu(x, value), 'b-', linewidth=2.5, label=f'GLU(gate, {value}) = sigmoid(gate)×{value}')
ax2.plot(x, swiglu(x, value), 'g-', linewidth=2.5, label=f'SwiGLU(gate, {value}) = swish(gate)×{value}')

ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlabel('Gate Input', fontsize=12)
ax2.set_ylabel('Output', fontsize=12)
ax2.set_title(f'Output when Value = {value}\n(Gate controls how much value passes)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

# --- Plot 3: Output comparison (value=-1) ---
ax3 = fig.add_subplot(2, 3, 3)
value = -1.0
ax3.plot(x, glu(x, value), 'b-', linewidth=2.5, label=f'GLU(gate, {value})')
ax3.plot(x, swiglu(x, value), 'g-', linewidth=2.5, label=f'SwiGLU(gate, {value})')

ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax3.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
ax3.set_xlim(-4, 4)
ax3.set_ylim(-1.5, 0.5)
ax3.set_xlabel('Gate Input', fontsize=12)
ax3.set_ylabel('Output', fontsize=12)
ax3.set_title(f'Output when Value = {value}\n(Negative value, gate still controls)', fontsize=14, fontweight='bold')
ax3.legend(loc='lower left', fontsize=10)
ax3.grid(True, alpha=0.3)

# --- Plot 4: 3D Surface - GLU ---
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
gate_range = np.linspace(-3, 3, 50)
value_range = np.linspace(-2, 2, 50)
G, V = np.meshgrid(gate_range, value_range)
Z_glu = glu(G, V)

surf1 = ax4.plot_surface(G, V, Z_glu, cmap='Blues', alpha=0.8, edgecolor='none')
ax4.set_xlabel('Gate', fontsize=10)
ax4.set_ylabel('Value', fontsize=10)
ax4.set_zlabel('Output', fontsize=10)
ax4.set_title('GLU: sigmoid(gate) × value', fontsize=14, fontweight='bold')
ax4.view_init(elev=25, azim=45)

# --- Plot 5: 3D Surface - SwiGLU ---
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
Z_swiglu = swiglu(G, V)

surf2 = ax5.plot_surface(G, V, Z_swiglu, cmap='Greens', alpha=0.8, edgecolor='none')
ax5.set_xlabel('Gate', fontsize=10)
ax5.set_ylabel('Value', fontsize=10)
ax5.set_zlabel('Output', fontsize=10)
ax5.set_title('SwiGLU: swish(gate) × value', fontsize=14, fontweight='bold')
ax5.view_init(elev=25, azim=45)

# --- Plot 6: Architecture comparison ---
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

# Draw architecture diagram
arch_text = """
┌─────────────────────────────────────────────────────────┐
│                  FFN Architecture Comparison            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Standard FFN (GPT-3):                                  │
│  ═══════════════════                                    │
│      x ──► W_up ──► GeLU ──► W_down ──► output         │
│           [d→4d]           [4d→d]                       │
│                                                         │
│  Parameters: 8d²                                        │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  SwiGLU FFN (LLaMA):                                    │
│  ══════════════════                                     │
│            ┌──► W_gate ──► Swish ──┐                   │
│      x ───┤                        ├──► × ──► W_down   │
│            └──► W_up ─────────────┘        [d'→d]      │
│                                                         │
│  Parameters: 3×d×d' ≈ 8d² when d' = 8d/3               │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Key Insight:                                           │
│  • GLU/SwiGLU = Gate × Value (element-wise multiply)   │
│  • Gate learns WHICH features to pass through          │
│  • Value learns WHAT information to pass               │
│  • This is like a "soft" Mixture of Experts!           │
│                                                         │
└─────────────────────────────────────────────────────────┘
"""

ax6.text(0.5, 0.5, arch_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='center', horizontalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax6.set_title('Architecture Comparison', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('glu_swiglu_comparison.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

print("Image saved: glu_swiglu_comparison.png")

# ============================================================
# Print key differences
# ============================================================
print("\n" + "=" * 60)
print("GLU vs SwiGLU Key Differences")
print("=" * 60)

print("""
┌────────────┬─────────────────────┬─────────────────────┐
│            │ GLU                 │ SwiGLU              │
├────────────┼─────────────────────┼─────────────────────┤
│ Formula    │ σ(xW_g) ⊙ (xW_v)   │ Swish(xW_g) ⊙ (xW_v)│
├────────────┼─────────────────────┼─────────────────────┤
│ Gate Range │ (0, 1)              │ (-∞, +∞)            │
├────────────┼─────────────────────┼─────────────────────┤
│ Gate at 0  │ σ(0) = 0.5          │ Swish(0) = 0        │
├────────────┼─────────────────────┼─────────────────────┤
│ Gradient   │ Limited by σ'       │ Better gradient flow│
├────────────┼─────────────────────┼─────────────────────┤
│ Used In    │ Early experiments   │ LLaMA, PaLM, etc.   │
└────────────┴─────────────────────┴─────────────────────┘
""")

print("\nWhy SwiGLU is better:")
print("1. Swish gate has unbounded range -> more expressive")
print("2. Non-monotonic gate allows negative contributions")
print("3. Smoother gradients -> better training stability")
print("4. Empirically: lower perplexity at same compute budget")
