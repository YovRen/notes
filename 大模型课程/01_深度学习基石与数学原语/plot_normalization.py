"""
Visualization of Normalization Techniques:
- BatchNorm vs LayerNorm vs RMSNorm
- Effect of gamma (scale) and beta (shift) parameters
- Pre-Norm vs Post-Norm gradient flow
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#e94560'
plt.rcParams['axes.labelcolor'] = '#eee'
plt.rcParams['text.color'] = '#eee'
plt.rcParams['xtick.color'] = '#eee'
plt.rcParams['ytick.color'] = '#eee'
plt.rcParams['grid.color'] = '#0f3460'
plt.rcParams['grid.alpha'] = 0.5

def layer_norm(x, gamma=1.0, beta=0.0, eps=1e-6):
    """LayerNorm: normalize over feature dimension"""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def rms_norm(x, gamma=1.0, eps=1e-6):
    """RMSNorm: no mean subtraction"""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return gamma * (x / rms)

def batch_norm(x, gamma=1.0, beta=0.0, eps=1e-6):
    """BatchNorm: normalize over batch dimension"""
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.35, wspace=0.3)

# ============================================================
# Part 1: Visualization of normalization dimensions
# ============================================================
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 12)
ax1.set_ylim(0, 5)
ax1.axis('off')
ax1.set_title('Normalization Dimension Comparison\n(Batch=4, Seq=3, Dim=5)', 
              fontsize=14, fontweight='bold', pad=10)

# Define colors
colors = {
    'bn': '#e94560',  # Red for BatchNorm
    'ln': '#00d9ff',  # Cyan for LayerNorm
    'rms': '#00ff88'  # Green for RMSNorm
}

def draw_tensor_grid(ax, x_start, y_start, title, highlight_type, width=3, height=3):
    """Draw a 3D-like tensor visualization"""
    cell_w, cell_h = 0.5, 0.5
    depth_offset = 0.15
    
    ax.text(x_start + width*cell_w/2, y_start + height*cell_h + 0.6, title, 
            ha='center', fontsize=11, fontweight='bold')
    
    # Draw 4 layers (batch dimension)
    for b in range(4):
        offset_x = b * depth_offset
        offset_y = -b * depth_offset * 0.5
        
        for i in range(3):  # seq
            for j in range(5):  # dim
                x = x_start + j * cell_w * 0.4 + offset_x
                y = y_start + (2-i) * cell_h * 0.5 + offset_y
                
                # Determine color based on highlight type
                if highlight_type == 'batch':
                    # BatchNorm: same color for same feature across batch
                    alpha = min(0.4 + 0.1 * b, 0.9)
                    color = plt.cm.Reds(0.3 + j * 0.12)
                elif highlight_type == 'layer':
                    # LayerNorm: same color for same token
                    alpha = min(0.4 + 0.1 * b, 0.9)
                    color = plt.cm.Blues(0.3 + (b * 3 + i) * 0.04)
                else:  # rms
                    alpha = min(0.4 + 0.1 * b, 0.9)
                    color = plt.cm.Greens(0.3 + (b * 3 + i) * 0.04)
                
                rect = FancyBboxPatch((x, y), cell_w*0.35, cell_h*0.4,
                                      boxstyle="round,pad=0.02",
                                      facecolor=color, edgecolor='white',
                                      linewidth=0.5, alpha=alpha)
                ax.add_patch(rect)
    
    # Add dimension labels
    ax.text(x_start - 0.2, y_start + height*cell_h*0.25, 'Seq', fontsize=8, rotation=90, va='center')
    ax.text(x_start + width*cell_w*0.4, y_start - 0.3, 'Dim', fontsize=8, ha='center')

draw_tensor_grid(ax1, 0.5, 1, 'BatchNorm\n(across batch, per feature)', 'batch')
draw_tensor_grid(ax1, 4.2, 1, 'LayerNorm\n(per token, across features)', 'layer')
draw_tensor_grid(ax1, 7.9, 1, 'RMSNorm\n(per token, no mean)', 'rms')

# Add formula annotations
formulas = [
    (2.0, 0.3, r'$\mu_j = \frac{1}{B}\sum_b x_{b,j}$', colors['bn']),
    (5.7, 0.3, r'$\mu = \frac{1}{d}\sum_i x_i$', colors['ln']),
    (9.4, 0.3, r'$\text{RMS} = \sqrt{\frac{1}{d}\sum x_i^2}$', colors['rms'])
]
for x, y, formula, color in formulas:
    ax1.text(x, y, formula, fontsize=10, ha='center', color=color,
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor=color, alpha=0.8))

# ============================================================
# Part 2: Distribution before/after normalization
# ============================================================
np.random.seed(42)

# Generate sample data with different scales per feature
x_raw = np.random.randn(1000, 64) * np.array([0.5 + i*0.1 for i in range(64)])
x_raw += np.array([-5 + i*0.2 for i in range(64)])  # Add offset

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Before Normalization\n(Different scales per feature)', fontsize=11, fontweight='bold')
for i in [0, 20, 40, 63]:
    ax2.hist(x_raw[:, i], bins=30, alpha=0.5, label=f'dim={i}', density=True)
ax2.set_xlabel('Value')
ax2.set_ylabel('Density')
ax2.legend(fontsize=8)
ax2.set_xlim(-20, 20)

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_title('After LayerNorm\n(mean=0, var=1 per sample)', fontsize=11, fontweight='bold')
x_ln = layer_norm(x_raw)
for i in [0, 20, 40, 63]:
    ax3.hist(x_ln[:, i], bins=30, alpha=0.5, label=f'dim={i}', density=True)
ax3.set_xlabel('Value')
ax3.legend(fontsize=8)
ax3.set_xlim(-4, 4)

ax4 = fig.add_subplot(gs[1, 2])
ax4.set_title('After RMSNorm\n(no mean centering)', fontsize=11, fontweight='bold')
x_rms = rms_norm(x_raw)
for i in [0, 20, 40, 63]:
    ax4.hist(x_rms[:, i], bins=30, alpha=0.5, label=f'dim={i}', density=True)
ax4.set_xlabel('Value')
ax4.legend(fontsize=8)
ax4.set_xlim(-4, 4)

# ============================================================
# Part 3: Effect of gamma and beta parameters
# ============================================================
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_title('Effect of $\\gamma$ (Scale)\nLayerNorm output', fontsize=11, fontweight='bold')

x_single = np.random.randn(500, 32)
gammas = [0.5, 1.0, 2.0]
colors_gamma = ['#ff6b6b', '#4ecdc4', '#45b7d1']

for gamma, color in zip(gammas, colors_gamma):
    x_out = layer_norm(x_single, gamma=gamma, beta=0)
    ax5.hist(x_out.flatten(), bins=50, alpha=0.5, label=f'γ={gamma}', 
             color=color, density=True)
ax5.set_xlabel('Value')
ax5.set_ylabel('Density')
ax5.legend()
ax5.axvline(0, color='white', linestyle='--', alpha=0.5)

ax6 = fig.add_subplot(gs[2, 1])
ax6.set_title('Effect of $\\beta$ (Shift)\nLayerNorm output', fontsize=11, fontweight='bold')

betas = [-1.0, 0.0, 1.0]
colors_beta = ['#ff6b6b', '#4ecdc4', '#45b7d1']

for beta, color in zip(betas, colors_beta):
    x_out = layer_norm(x_single, gamma=1.0, beta=beta)
    ax6.hist(x_out.flatten(), bins=50, alpha=0.5, label=f'β={beta}', 
             color=color, density=True)
ax6.set_xlabel('Value')
ax6.legend()
for beta, color in zip(betas, colors_beta):
    ax6.axvline(beta, color=color, linestyle='--', alpha=0.7)

# ============================================================
# Part 4: LayerNorm vs RMSNorm numerical comparison
# ============================================================
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_title('LayerNorm vs RMSNorm\n(Same input, different outputs)', fontsize=11, fontweight='bold')

# Create input with non-zero mean
x_test = np.random.randn(1, 64) + 3  # shift mean to 3
x_ln_out = layer_norm(x_test).flatten()
x_rms_out = rms_norm(x_test).flatten()

dims = np.arange(64)
width = 0.35
ax7.bar(dims - width/2, x_ln_out, width, label='LayerNorm', color='#00d9ff', alpha=0.7)
ax7.bar(dims + width/2, x_rms_out, width, label='RMSNorm', color='#00ff88', alpha=0.7)
ax7.set_xlabel('Feature Dimension')
ax7.set_ylabel('Normalized Value')
ax7.legend()
ax7.set_xlim(-1, 64)

# Add annotation about the difference
ax7.annotate('RMSNorm: no mean subtraction\n→ preserves relative offset',
             xy=(32, max(x_rms_out)), xytext=(45, max(x_rms_out)+0.5),
             fontsize=9, color='#00ff88',
             arrowprops=dict(arrowstyle='->', color='#00ff88'))

plt.suptitle('Normalization Techniques Comparison', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()

print("Saved: normalization_comparison.png")

# ============================================================
# Figure 2: Pre-Norm vs Post-Norm Architecture
# ============================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 8))

def draw_block(ax, x, y, w, h, text, color, fontsize=9):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', color='white')

def draw_arrow(ax, start, end, color='white'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

def draw_residual(ax, x1, y1, x2, y2, color='#ffd93d'):
    """Draw curved residual connection"""
    from matplotlib.patches import FancyArrowPatch
    import matplotlib.patches as mpatches
    
    # Draw a curved line
    mid_x = x1 - 0.8
    ax.plot([x1, mid_x, mid_x, x2], [y1, y1, y2, y2], 
            color=color, linewidth=2, linestyle='--')
    ax.annotate('', xy=(x2, y2), xytext=(mid_x, y2),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))

# Post-Norm architecture
ax_post = axes[0]
ax_post.set_xlim(-2, 6)
ax_post.set_ylim(0, 10)
ax_post.axis('off')
ax_post.set_title('Post-Norm (Original Transformer)\n$x_{out} = Norm(x + Sublayer(x))$', 
                  fontsize=12, fontweight='bold')

# Draw blocks
draw_block(ax_post, 1, 0.5, 2.5, 0.8, 'Input x', '#0f3460', fontsize=10)
draw_arrow(ax_post, (2.25, 1.3), (2.25, 2))
draw_block(ax_post, 1, 2, 2.5, 1.2, 'Sublayer\n(Attn/FFN)', '#e94560')
draw_arrow(ax_post, (2.25, 3.2), (2.25, 3.8))
draw_block(ax_post, 1, 3.8, 2.5, 0.8, 'Add (+)', '#4a4a6a')
draw_arrow(ax_post, (2.25, 4.6), (2.25, 5.2))
draw_block(ax_post, 1, 5.2, 2.5, 1.0, 'LayerNorm', '#00d9ff')
draw_arrow(ax_post, (2.25, 6.2), (2.25, 7))
draw_block(ax_post, 1, 7, 2.5, 0.8, 'Output', '#0f3460', fontsize=10)

# Residual connection
draw_residual(ax_post, 1, 1.3, 1, 4.2)
ax_post.text(-0.5, 2.8, 'Residual\nPath', fontsize=9, color='#ffd93d', ha='center')

# Gradient flow annotation
ax_post.annotate('Gradient must\nflow through Norm', 
                 xy=(3.5, 5.7), xytext=(4.5, 6.5),
                 fontsize=9, color='#ff6b6b',
                 arrowprops=dict(arrowstyle='->', color='#ff6b6b'))

# Pre-Norm architecture
ax_pre = axes[1]
ax_pre.set_xlim(-2, 6)
ax_pre.set_ylim(0, 10)
ax_pre.axis('off')
ax_pre.set_title('Pre-Norm (LLaMA/GPT-2)\n$x_{out} = x + Sublayer(Norm(x))$', 
                 fontsize=12, fontweight='bold')

# Draw blocks
draw_block(ax_pre, 1, 0.5, 2.5, 0.8, 'Input x', '#0f3460', fontsize=10)
draw_arrow(ax_pre, (2.25, 1.3), (2.25, 2))
draw_block(ax_pre, 1, 2, 2.5, 1.0, 'RMSNorm', '#00ff88')
draw_arrow(ax_pre, (2.25, 3), (2.25, 3.6))
draw_block(ax_pre, 1, 3.6, 2.5, 1.2, 'Sublayer\n(Attn/FFN)', '#e94560')
draw_arrow(ax_pre, (2.25, 4.8), (2.25, 5.4))
draw_block(ax_pre, 1, 5.4, 2.5, 0.8, 'Add (+)', '#4a4a6a')
draw_arrow(ax_pre, (2.25, 6.2), (2.25, 7))
draw_block(ax_pre, 1, 7, 2.5, 0.8, 'Output', '#0f3460', fontsize=10)

# Residual connection (identity path)
draw_residual(ax_pre, 1, 1.3, 1, 5.8)
ax_pre.text(-0.5, 3.5, 'Identity\nPath', fontsize=9, color='#ffd93d', ha='center')

# Gradient flow annotation
ax_pre.annotate('Gradient can flow\ndirectly (identity)', 
                xy=(-0.3, 4.5), xytext=(-1.5, 5.5),
                fontsize=9, color='#00ff88',
                arrowprops=dict(arrowstyle='->', color='#00ff88'))

# Add comparison box
comparison_text = """
Key Differences:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Post-Norm:
  • Gradient must pass through Norm layer
  • Needs careful warmup for deep networks
  • Better final representation (theory)

Pre-Norm:
  • Identity path for gradient flow
  • Much more stable for very deep models
  • Used by LLaMA, GPT-2, PaLM, DeepSeek
"""

fig2.text(0.5, 0.02, comparison_text, ha='center', va='bottom', fontsize=10,
         family='monospace', color='#aaa',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#444'))

plt.tight_layout(rect=[0, 0.15, 1, 0.95])
plt.savefig('prenorm_vs_postnorm.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()

print("Saved: prenorm_vs_postnorm.png")

# ============================================================
# Figure 3: Gradient magnitude through layers
# ============================================================
fig3, (ax_grad, ax_formula) = plt.subplots(1, 2, figsize=(14, 5))

# Simulate gradient norms through layers
np.random.seed(123)
n_layers = 32

# Post-Norm: gradient can decay/explode
grad_post = [1.0]
for i in range(n_layers - 1):
    # Gradient through norm + sublayer can have varying scale
    scale = np.random.uniform(0.85, 1.05)  
    grad_post.append(grad_post[-1] * scale)

# Pre-Norm: gradient has identity path
grad_pre = [1.0]
for i in range(n_layers - 1):
    # Identity path preserves gradient + some contribution from sublayer
    grad_pre.append(1.0 + np.random.uniform(-0.1, 0.1))

layers = np.arange(1, n_layers + 1)

ax_grad.plot(layers, grad_post, 'o-', color='#e94560', label='Post-Norm', 
             linewidth=2, markersize=4)
ax_grad.plot(layers, grad_pre, 's-', color='#00ff88', label='Pre-Norm', 
             linewidth=2, markersize=4)
ax_grad.axhline(1.0, color='white', linestyle='--', alpha=0.5, label='Ideal')
ax_grad.fill_between(layers, 0.8, 1.2, alpha=0.2, color='#00ff88', label='Stable zone')

ax_grad.set_xlabel('Layer (backward from output)', fontsize=11)
ax_grad.set_ylabel('Gradient Norm (relative)', fontsize=11)
ax_grad.set_title('Gradient Flow Stability\n(Simulated for 32-layer model)', 
                  fontsize=12, fontweight='bold')
ax_grad.legend(loc='upper left')
ax_grad.set_ylim(0, 2)
ax_grad.set_xlim(1, 32)

# Formula comparison
ax_formula.axis('off')
ax_formula.set_title('Mathematical Comparison', fontsize=12, fontweight='bold')

formulas_text = """
┌─────────────────────────────────────────────────────────────────┐
│  LayerNorm                                                      │
│  ══════════                                                     │
│                                                                 │
│  $\\mu = \\frac{1}{d}\\sum_{i=1}^d x_i$                            │
│                                                                 │
│  $\\sigma^2 = \\frac{1}{d}\\sum_{i=1}^d (x_i - \\mu)^2$            │
│                                                                 │
│  $y_i = \\gamma \\frac{x_i - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta$  │
│                                                                 │
│  Parameters: $\\gamma, \\beta \\in \\mathbb{R}^d$ (2d params)       │
├─────────────────────────────────────────────────────────────────┤
│  RMSNorm                                                        │
│  ═══════                                                        │
│                                                                 │
│  $\\text{RMS}(x) = \\sqrt{\\frac{1}{d}\\sum_{i=1}^d x_i^2 + \\epsilon}$│
│                                                                 │
│  $y_i = \\gamma \\frac{x_i}{\\text{RMS}(x)}$                        │
│                                                                 │
│  Parameters: $\\gamma \\in \\mathbb{R}^d$ (d params, no $\\beta$)   │
│                                                                 │
│  ✓ No mean computation (faster)                                 │
│  ✓ No bias term (simpler)                                       │
│  ✓ 10-40% faster with fusion                                    │
└─────────────────────────────────────────────────────────────────┘
"""

ax_formula.text(0.5, 0.5, formulas_text, ha='center', va='center',
               fontsize=11, family='monospace',
               transform=ax_formula.transAxes,
               bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#00d9ff', pad=0.5))

plt.tight_layout()
plt.savefig('gradient_flow_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()

print("Saved: gradient_flow_comparison.png")
print("\n✅ All normalization visualizations completed!")
