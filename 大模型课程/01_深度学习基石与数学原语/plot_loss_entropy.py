"""
Visualization of Loss Functions and Entropy Concepts:
1. Softmax: logits → probabilities
2. NLLLoss: -log(p)
3. CrossEntropy = Softmax + NLLLoss
4. Entropy, Cross-Entropy, KL Divergence relationship
5. Gradient comparison: MSE vs CrossEntropy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.sankey import Sankey

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#444'
plt.rcParams['axes.labelcolor'] = '#eee'
plt.rcParams['text.color'] = '#eee'
plt.rcParams['xtick.color'] = '#eee'
plt.rcParams['ytick.color'] = '#eee'
plt.rcParams['grid.color'] = '#0f3460'
plt.rcParams['grid.alpha'] = 0.5

# ============================================================
# Figure 1: The Pipeline - Logits → Softmax → NLLLoss
# ============================================================
fig1 = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.4, wspace=0.3)

# Top row: The pipeline diagram
ax_pipe = fig1.add_subplot(gs[0, :])
ax_pipe.set_xlim(0, 16)
ax_pipe.set_ylim(0, 6)
ax_pipe.axis('off')
ax_pipe.set_title('The Pipeline: Logits → Softmax → Probability → NLLLoss → Loss', 
                  fontsize=14, fontweight='bold', pad=15)

def draw_box(ax, x, y, w, h, text, color, fontsize=10):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', color='white')

def draw_arrow(ax, x1, y1, x2, y2, text='', color='white'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.3, text, ha='center', fontsize=9, color=color)

# Draw the pipeline boxes
# Logits
draw_box(ax_pipe, 0.5, 2, 2.5, 2, 'Logits\n[2.0, 4.5, 1.0]', '#e94560')
draw_arrow(ax_pipe, 3, 3, 4, 3)

# Softmax
draw_box(ax_pipe, 4, 2, 2.5, 2, 'Softmax\n$\\frac{e^{z_i}}{\\sum e^{z_j}}$', '#ffd93d')
draw_arrow(ax_pipe, 6.5, 3, 7.5, 3)

# Probabilities
draw_box(ax_pipe, 7.5, 2, 2.5, 2, 'Probs\n[0.08, 0.89, 0.03]', '#00d9ff')
draw_arrow(ax_pipe, 10, 3, 11, 3)

# NLLLoss
draw_box(ax_pipe, 11, 2, 2.5, 2, 'NLLLoss\n$-\\log(p_{correct})$', '#00ff88')
draw_arrow(ax_pipe, 13.5, 3, 14.5, 3)

# Loss value
draw_box(ax_pipe, 14.5, 2.3, 1.2, 1.4, 'Loss\n0.12', '#9b59b6')

# Target annotation
ax_pipe.annotate('Target: class 1 ("blue")', xy=(8.75, 2), xytext=(8.75, 0.8),
                fontsize=10, color='#00d9ff', ha='center',
                arrowprops=dict(arrowstyle='->', color='#00d9ff'))

# CrossEntropyLoss bracket
ax_pipe.plot([4, 4], [4.3, 4.6], 'w-', lw=2)
ax_pipe.plot([4, 13.5], [4.6, 4.6], 'w-', lw=2)
ax_pipe.plot([13.5, 13.5], [4.3, 4.6], 'w-', lw=2)
ax_pipe.text(8.75, 5, 'nn.CrossEntropyLoss (does both!)', ha='center', 
             fontsize=11, fontweight='bold', color='#ff6b6b')

# ============================================================
# Bottom left: -log(p) curve
# ============================================================
ax_nll = fig1.add_subplot(gs[1, 0])

p = np.linspace(0.001, 1, 500)
nll = -np.log(p)

ax_nll.plot(p, nll, color='#00ff88', linewidth=3, label='$-\\log(p)$')
ax_nll.fill_between(p, nll, alpha=0.2, color='#00ff88')

# Mark key points
points = [(0.01, -np.log(0.01)), (0.1, -np.log(0.1)), 
          (0.5, -np.log(0.5)), (0.9, -np.log(0.9)), (0.99, -np.log(0.99))]
for px, py in points:
    ax_nll.plot(px, py, 'o', color='#e94560', markersize=8)
    ax_nll.annotate(f'p={px}\nLoss={py:.2f}', xy=(px, py), 
                   xytext=(px+0.1, py+0.3), fontsize=8,
                   arrowprops=dict(arrowstyle='->', color='#e94560', lw=1))

ax_nll.set_xlabel('Predicted Probability (for correct class)', fontsize=11)
ax_nll.set_ylabel('Loss = $-\\log(p)$', fontsize=11)
ax_nll.set_title('NLLLoss: The "-log(p)" Curve\n(Lower probability → Higher loss)', 
                 fontsize=12, fontweight='bold')
ax_nll.set_xlim(0, 1.1)
ax_nll.set_ylim(0, 5)
ax_nll.legend(fontsize=10)
ax_nll.axhline(0, color='white', linestyle='--', alpha=0.3)

# ============================================================
# Bottom right: Softmax visualization
# ============================================================
ax_soft = fig1.add_subplot(gs[1, 1])

logits = np.array([2.0, 4.5, 1.0])
exp_logits = np.exp(logits)
probs = exp_logits / exp_logits.sum()

x = np.arange(3)
width = 0.35

# Bar chart
bars1 = ax_soft.bar(x - width/2, logits, width, label='Logits (raw)', color='#e94560', alpha=0.8)
bars2 = ax_soft.bar(x + width/2, probs, width, label='Softmax (prob)', color='#00d9ff', alpha=0.8)

# Add value labels
for bar, val in zip(bars1, logits):
    ax_soft.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', fontsize=9, color='#e94560')
for bar, val in zip(bars2, probs):
    ax_soft.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', fontsize=9, color='#00d9ff')

ax_soft.set_xticks(x)
ax_soft.set_xticklabels(['green (0)', 'blue (1)', 'red (2)'])
ax_soft.set_ylabel('Value', fontsize=11)
ax_soft.set_title('Softmax: Logits → Probabilities\n(Sum of probs = 1.0)', 
                  fontsize=12, fontweight='bold')
ax_soft.legend(loc='upper right')

# Highlight the correct class
ax_soft.annotate('Target!', xy=(1 + width/2, probs[1]), xytext=(1.5, probs[1] + 0.15),
                fontsize=10, color='#00ff88', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=2))

plt.tight_layout()
plt.savefig('softmax_nll_pipeline.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("Saved: softmax_nll_pipeline.png")

# ============================================================
# Figure 2: Entropy Concepts Visualization
# ============================================================
fig2 = plt.figure(figsize=(16, 10))
gs2 = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

# 2a: What is Entropy?
ax_ent = fig2.add_subplot(gs2[0, 0])

# Different distributions with different entropy
dists = {
    'Uniform [0.25, 0.25, 0.25, 0.25]': [0.25, 0.25, 0.25, 0.25],
    'Peaked [0.7, 0.1, 0.1, 0.1]': [0.7, 0.1, 0.1, 0.1],
    'One-Hot [1, 0, 0, 0]': [0.999, 0.0003, 0.0003, 0.0004],  # avoid log(0)
}

x = np.arange(4)
width = 0.25
colors = ['#e94560', '#ffd93d', '#00d9ff']

for i, (name, dist) in enumerate(dists.items()):
    dist = np.array(dist)
    entropy = -np.sum(dist * np.log(dist + 1e-10))
    bars = ax_ent.bar(x + i*width - width, dist, width, 
                      label=f'{name}\nEntropy={entropy:.2f}', color=colors[i], alpha=0.8)

ax_ent.set_xticks(x)
ax_ent.set_xticklabels(['A', 'B', 'C', 'D'])
ax_ent.set_ylabel('Probability', fontsize=11)
ax_ent.set_title('Entropy: Measure of "Uncertainty"\n$H(P) = -\\sum P(x) \\log P(x)$', 
                 fontsize=12, fontweight='bold')
ax_ent.legend(fontsize=8, loc='upper right')
ax_ent.set_ylim(0, 1.1)

# Add annotation
ax_ent.text(0.5, 0.95, 'Higher entropy = More uncertain\nLower entropy = More confident',
           transform=ax_ent.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#00ff88'))

# 2b: Cross-Entropy vs KL Divergence
ax_kl = fig2.add_subplot(gs2[0, 1])
ax_kl.axis('off')

# Draw Venn-like diagram
circle1 = plt.Circle((0.35, 0.5), 0.25, fill=False, edgecolor='#00d9ff', linewidth=3)
circle2 = plt.Circle((0.65, 0.5), 0.25, fill=False, edgecolor='#e94560', linewidth=3)
ax_kl.add_patch(circle1)
ax_kl.add_patch(circle2)

ax_kl.text(0.35, 0.5, '$H(P)$\nEntropy\nof Target', ha='center', va='center', 
           fontsize=10, color='#00d9ff')
ax_kl.text(0.65, 0.5, '$D_{KL}(P||Q)$\nKL Divergence', ha='center', va='center', 
           fontsize=10, color='#e94560')

# Cross-entropy bracket
ax_kl.plot([0.1, 0.1, 0.9, 0.9], [0.15, 0.1, 0.1, 0.15], 'w-', lw=2)
ax_kl.text(0.5, 0.02, 'Cross-Entropy $H(P, Q) = H(P) + D_{KL}(P||Q)$', 
           ha='center', fontsize=11, fontweight='bold', color='#00ff88')

ax_kl.set_title('Relationship: Entropy, Cross-Entropy, KL Divergence', 
                fontsize=12, fontweight='bold')
ax_kl.set_xlim(0, 1)
ax_kl.set_ylim(0, 1)

# Formula box
formula_text = """
For One-Hot Target (LLM case):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• $H(P)$ = 0  (no uncertainty in target)
• $H(P,Q)$ = $D_{KL}(P||Q)$ = $-\\log(q_{correct})$

So minimizing Cross-Entropy 
= minimizing KL Divergence
= making model distribution match target!
"""
ax_kl.text(0.5, 0.75, formula_text, ha='center', va='center', fontsize=9,
          bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#ffd93d', pad=0.5))

# 2c: Gradient comparison - MSE vs CrossEntropy
ax_grad = fig2.add_subplot(gs2[1, 0])

p = np.linspace(0.01, 0.99, 100)

# MSE gradient: 2(p-1) * p(1-p)
grad_mse = np.abs(2 * (p - 1) * p * (1 - p))

# CrossEntropy gradient: |p - 1|
grad_ce = np.abs(p - 1)

ax_grad.plot(p, grad_mse, color='#e94560', linewidth=3, label='MSE gradient: $|2(p-1) \\cdot p(1-p)|$')
ax_grad.plot(p, grad_ce, color='#00ff88', linewidth=3, label='CrossEntropy gradient: $|p-1|$')

ax_grad.fill_between(p, grad_mse, alpha=0.2, color='#e94560')
ax_grad.fill_between(p, grad_ce, alpha=0.2, color='#00ff88')

# Highlight the problem area
ax_grad.axvspan(0, 0.2, alpha=0.3, color='#ffd93d', label='Problem zone (p≈0)')
ax_grad.annotate('MSE: tiny gradient\nwhen p≈0!', xy=(0.05, 0.02), xytext=(0.25, 0.3),
                fontsize=9, color='#e94560',
                arrowprops=dict(arrowstyle='->', color='#e94560'))
ax_grad.annotate('CE: large gradient\nwhen p≈0!', xy=(0.05, 0.95), xytext=(0.25, 0.7),
                fontsize=9, color='#00ff88',
                arrowprops=dict(arrowstyle='->', color='#00ff88'))

ax_grad.set_xlabel('Predicted probability p (for correct class)', fontsize=11)
ax_grad.set_ylabel('|Gradient| (absolute value)', fontsize=11)
ax_grad.set_title('Why CrossEntropy? Better Gradients!\n(When prediction is wrong, gradient should be LARGE)', 
                  fontsize=12, fontweight='bold')
ax_grad.legend(fontsize=9, loc='upper right')
ax_grad.set_ylim(0, 1.1)

# 2d: Perplexity visualization
ax_ppl = fig2.add_subplot(gs2[1, 1])

# Different loss values and their perplexity
losses = np.linspace(0, 4, 100)
ppls = np.exp(losses)

ax_ppl.plot(losses, ppls, color='#9b59b6', linewidth=3)
ax_ppl.fill_between(losses, ppls, alpha=0.2, color='#9b59b6')

# Mark key points
key_losses = [0, 1, 2, 3]
for loss in key_losses:
    ppl = np.exp(loss)
    ax_ppl.plot(loss, ppl, 'o', color='#00ff88', markersize=10)
    ax_ppl.annotate(f'Loss={loss}\nPPL={ppl:.1f}', xy=(loss, ppl), 
                   xytext=(loss+0.2, ppl+2), fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='#00ff88'))

ax_ppl.set_xlabel('Cross-Entropy Loss', fontsize=11)
ax_ppl.set_ylabel('Perplexity (PPL)', fontsize=11)
ax_ppl.set_title('Perplexity = $e^{Loss}$\n(How many choices is model "confused" between?)', 
                 fontsize=12, fontweight='bold')

# Add interpretation
interp_text = """
PPL = 1: Perfect (no confusion)
PPL = 10: Like guessing among 10 words
PPL = 100: Very confused
"""
ax_ppl.text(0.95, 0.95, interp_text, transform=ax_ppl.transAxes, 
           fontsize=9, va='top', ha='right',
           bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#9b59b6'))

plt.tight_layout()
plt.savefig('entropy_concepts.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("Saved: entropy_concepts.png")

# ============================================================
# Figure 3: Interactive-style comparison table
# ============================================================
fig3, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Title
ax.text(0.5, 0.95, 'Summary: Loss Functions & Entropy Concepts', 
        ha='center', va='top', fontsize=16, fontweight='bold',
        transform=ax.transAxes)

# Create a detailed summary table
summary = """
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CONCEPT RELATIONSHIPS                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│   LOGITS ──────────► SOFTMAX ──────────► PROBABILITY ──────────► NLLLoss ──────► LOSS     │
│   [2.0, 4.5, 1.0]    exp & normalize    [0.08, 0.89, 0.03]       -log(p)        0.12      │
│                                                                                             │
│   ◄─────────────────── nn.CrossEntropyLoss (combines both) ───────────────────►           │
│                                                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│   ENTROPY H(P)              = "Uncertainty in distribution P"                               │
│                             = $-\\sum P(x) \\log P(x)$                                        │
│                             • Uniform dist → High entropy (very uncertain)                 │
│                             • One-hot dist → Zero entropy (completely certain)             │
│                                                                                             │
│   CROSS-ENTROPY H(P,Q)      = "Cost of encoding P using Q's distribution"                  │
│                             = $-\\sum P(x) \\log Q(x)$                                        │
│                             • When P is one-hot: $H(P,Q) = -\\log(q_{correct})$              │
│                                                                                             │
│   KL DIVERGENCE D(P||Q)     = "Distance between P and Q"                                   │
│                             = $H(P,Q) - H(P)$                                               │
│                             • Always ≥ 0                                                   │
│                             • = 0 only when P = Q                                          │
│                                                                                             │
│   PERPLEXITY                = $e^{CrossEntropy}$                                            │
│                             = "Effective vocabulary size model is choosing from"           │
│                             • Lower is better (PPL=1 means perfect prediction)             │
│                                                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│   WHY CrossEntropy FOR CLASSIFICATION?                                                      │
│                                                                                             │
│   ✗ MSE:  Gradient ∝ p(1-p) → vanishes when p≈0 or p≈1                                    │
│   ✓ CE:   Gradient = (p-1)  → stays large when prediction is wrong                        │
│                                                                                             │
│   When model predicts p=0.01 for correct class:                                            │
│       MSE gradient = 0.02  (tiny! model can't learn)                                       │
│       CE gradient  = 0.99  (large! model learns fast)                                      │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
"""

ax.text(0.5, 0.45, summary, ha='center', va='center', fontsize=10,
       family='monospace', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#00d9ff', pad=0.5))

plt.savefig('loss_summary.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("Saved: loss_summary.png")

print("\n✅ All loss/entropy visualizations completed!")
