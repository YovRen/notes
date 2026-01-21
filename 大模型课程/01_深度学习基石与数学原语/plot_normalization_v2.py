"""
Visualization of Normalization Techniques v2
- Heatmap to show BEFORE and AFTER normalization
- Clearly show which dimension becomes "aligned"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.edgecolor'] = '#444'
plt.rcParams['axes.labelcolor'] = '#eee'
plt.rcParams['text.color'] = '#eee'
plt.rcParams['xtick.color'] = '#eee'
plt.rcParams['ytick.color'] = '#eee'

def layer_norm(x, eps=1e-6):
    """LayerNorm: normalize over feature dimension (axis=-1)"""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def rms_norm(x, eps=1e-6):
    """RMSNorm: no mean subtraction, only scale"""
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    return x / rms

def batch_norm(x, eps=1e-6):
    """BatchNorm: normalize over batch dimension (axis=0)"""
    mean = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

# ============================================================
# Create sample data with clear patterns
# ============================================================
np.random.seed(42)

# Shape: [Batch=8, Features=16]
# Each sample (row) has different mean and scale
# Each feature (column) also has different mean and scale
batch_size = 8
feature_dim = 16

# Create data where:
# - Different rows have different overall magnitudes (sample variation)
# - Different columns have different overall magnitudes (feature variation)
row_scales = np.array([0.5, 1.0, 2.0, 3.0, 1.5, 2.5, 0.8, 1.2]).reshape(-1, 1)
row_offsets = np.array([-2, 0, 3, -1, 2, 1, -3, 4]).reshape(-1, 1)
col_scales = np.linspace(0.5, 2.0, feature_dim).reshape(1, -1)
col_offsets = np.linspace(-3, 3, feature_dim).reshape(1, -1)

# Base random data + row variation + column variation
base_data = np.random.randn(batch_size, feature_dim) * 0.3
x_raw = base_data * row_scales * col_scales + row_offsets + col_offsets

print("Raw data statistics:")
print(f"  Row (sample) means: {x_raw.mean(axis=1).round(2)}")  # Each sample's mean
print(f"  Col (feature) means: {x_raw.mean(axis=0).round(2)}")  # Each feature's mean

# Apply normalizations
x_ln = layer_norm(x_raw)
x_bn = batch_norm(x_raw)
x_rms = rms_norm(x_raw)

# ============================================================
# Figure 1: Main comparison heatmap
# ============================================================
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], hspace=0.4, wspace=0.3)

# Shared colormap range for fair comparison
vmin, vmax = -4, 6

def plot_heatmap(ax, data, title, highlight_dim=None, show_stats=True):
    """Plot heatmap with optional row/column highlight"""
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Feature Dimension', fontsize=10)
    ax.set_ylabel('Sample (Batch)', fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    
    # Highlight which dimension is normalized
    if highlight_dim == 'row':
        # Highlight that each ROW is normalized (LayerNorm)
        for i in range(data.shape[0]):
            ax.add_patch(plt.Rectangle((-0.5, i-0.5), data.shape[1], 1, 
                                       fill=False, edgecolor='#00ff88', linewidth=2))
    elif highlight_dim == 'col':
        # Highlight that each COLUMN is normalized (BatchNorm)
        for j in range(data.shape[1]):
            ax.add_patch(plt.Rectangle((j-0.5, -0.5), 1, data.shape[0], 
                                       fill=False, edgecolor='#00ff88', linewidth=2))
    
    return im

# Row 1: Heatmaps
ax1 = fig.add_subplot(gs[0, 0])
plot_heatmap(ax1, x_raw, 'BEFORE Normalization\n(Raw Data)')

ax2 = fig.add_subplot(gs[0, 1])
plot_heatmap(ax2, x_bn, 'After BatchNorm\n(Each COLUMN normalized)', highlight_dim='col')

ax3 = fig.add_subplot(gs[0, 2])
plot_heatmap(ax3, x_ln, 'After LayerNorm\n(Each ROW normalized)', highlight_dim='row')

ax4 = fig.add_subplot(gs[0, 3])
plot_heatmap(ax4, x_rms, 'After RMSNorm\n(Each ROW scaled, no mean shift)', highlight_dim='row')

# ============================================================
# Row 2: Statistics comparison (bar charts)
# ============================================================

# 2a: Row means (sample means)
ax5 = fig.add_subplot(gs[1, 0])
x_pos = np.arange(batch_size)
width = 0.2
ax5.bar(x_pos - 1.5*width, x_raw.mean(axis=1), width, label='Raw', color='#e94560', alpha=0.8)
ax5.bar(x_pos - 0.5*width, x_bn.mean(axis=1), width, label='BatchNorm', color='#ffd93d', alpha=0.8)
ax5.bar(x_pos + 0.5*width, x_ln.mean(axis=1), width, label='LayerNorm', color='#00d9ff', alpha=0.8)
ax5.bar(x_pos + 1.5*width, x_rms.mean(axis=1), width, label='RMSNorm', color='#00ff88', alpha=0.8)
ax5.axhline(0, color='white', linestyle='--', alpha=0.5)
ax5.set_xlabel('Sample Index', fontsize=10)
ax5.set_ylabel('Mean', fontsize=10)
ax5.set_title('Row Means (per sample)\nLayerNorm → all rows have mean≈0', fontsize=11, fontweight='bold')
ax5.legend(fontsize=8, loc='upper right')
ax5.set_xticks(x_pos)

# 2b: Row stds
ax6 = fig.add_subplot(gs[1, 1])
ax6.bar(x_pos - 1.5*width, x_raw.std(axis=1), width, label='Raw', color='#e94560', alpha=0.8)
ax6.bar(x_pos - 0.5*width, x_bn.std(axis=1), width, label='BatchNorm', color='#ffd93d', alpha=0.8)
ax6.bar(x_pos + 0.5*width, x_ln.std(axis=1), width, label='LayerNorm', color='#00d9ff', alpha=0.8)
ax6.bar(x_pos + 1.5*width, x_rms.std(axis=1), width, label='RMSNorm', color='#00ff88', alpha=0.8)
ax6.axhline(1, color='white', linestyle='--', alpha=0.5, label='Target=1')
ax6.set_xlabel('Sample Index', fontsize=10)
ax6.set_ylabel('Std Dev', fontsize=10)
ax6.set_title('Row Stds (per sample)\nLayerNorm → all rows have std≈1', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8, loc='upper right')
ax6.set_xticks(x_pos)

# 2c: Column means (feature means)
ax7 = fig.add_subplot(gs[1, 2])
x_pos_col = np.arange(feature_dim)
ax7.bar(x_pos_col - 1.5*width, x_raw.mean(axis=0), width, label='Raw', color='#e94560', alpha=0.8)
ax7.bar(x_pos_col - 0.5*width, x_bn.mean(axis=0), width, label='BatchNorm', color='#ffd93d', alpha=0.8)
ax7.bar(x_pos_col + 0.5*width, x_ln.mean(axis=0), width, label='LayerNorm', color='#00d9ff', alpha=0.8)
ax7.bar(x_pos_col + 1.5*width, x_rms.mean(axis=0), width, label='RMSNorm', color='#00ff88', alpha=0.8)
ax7.axhline(0, color='white', linestyle='--', alpha=0.5)
ax7.set_xlabel('Feature Index', fontsize=10)
ax7.set_ylabel('Mean', fontsize=10)
ax7.set_title('Column Means (per feature)\nBatchNorm → all cols have mean≈0', fontsize=11, fontweight='bold')
ax7.legend(fontsize=8, loc='upper left')
ax7.set_xticks(x_pos_col[::2])

# 2d: Column stds
ax8 = fig.add_subplot(gs[1, 3])
ax8.bar(x_pos_col - 1.5*width, x_raw.std(axis=0), width, label='Raw', color='#e94560', alpha=0.8)
ax8.bar(x_pos_col - 0.5*width, x_bn.std(axis=0), width, label='BatchNorm', color='#ffd93d', alpha=0.8)
ax8.bar(x_pos_col + 0.5*width, x_ln.std(axis=0), width, label='LayerNorm', color='#00d9ff', alpha=0.8)
ax8.bar(x_pos_col + 1.5*width, x_rms.std(axis=0), width, label='RMSNorm', color='#00ff88', alpha=0.8)
ax8.axhline(1, color='white', linestyle='--', alpha=0.5, label='Target=1')
ax8.set_xlabel('Feature Index', fontsize=10)
ax8.set_ylabel('Std Dev', fontsize=10)
ax8.set_title('Column Stds (per feature)\nBatchNorm → all cols have std≈1', fontsize=11, fontweight='bold')
ax8.legend(fontsize=8, loc='upper left')
ax8.set_xticks(x_pos_col[::2])

plt.suptitle('Normalization: Which Dimension Gets "Aligned"?\n' + 
             'Green boxes show the normalization direction', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('normalization_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("Saved: normalization_heatmap.png")

# ============================================================
# Figure 2: Simplified 3D-like view showing normalization direction
# ============================================================
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

def draw_3d_tensor(ax, title, norm_direction, color):
    """Draw a simplified 3D tensor to show normalization direction"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Draw grid cells
    cell_w, cell_h = 0.8, 0.6
    n_rows, n_cols = 5, 8
    start_x, start_y = 1, 1.5
    
    for i in range(n_rows):
        for j in range(n_cols):
            x = start_x + j * cell_w
            y = start_y + (n_rows - 1 - i) * cell_h
            
            # Color based on normalization direction
            if norm_direction == 'row':
                # LayerNorm: same row has same hue
                c = plt.cm.Blues(0.3 + i * 0.12)
            elif norm_direction == 'col':
                # BatchNorm: same column has same hue
                c = plt.cm.Oranges(0.3 + j * 0.08)
            else:
                # Raw: random colors showing variation
                c = plt.cm.Reds(0.2 + np.random.rand() * 0.5)
            
            rect = plt.Rectangle((x, y), cell_w*0.9, cell_h*0.85, 
                                  facecolor=c, edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)
    
    # Add axis labels
    ax.text(start_x + n_cols*cell_w/2, start_y - 0.5, 'Feature Dimension (d)', 
            ha='center', fontsize=10)
    ax.text(start_x - 0.5, start_y + n_rows*cell_h/2, 'Samples\n(Batch)', 
            ha='center', va='center', fontsize=10, rotation=90)
    
    # Draw arrows showing normalization direction
    if norm_direction == 'row':
        # Arrow pointing horizontally (across features)
        for i in range(n_rows):
            y = start_y + (n_rows - 1 - i) * cell_h + cell_h * 0.4
            ax.annotate('', xy=(start_x + n_cols*cell_w + 0.3, y), 
                       xytext=(start_x + n_cols*cell_w + 1.2, y),
                       arrowprops=dict(arrowstyle='<-', color='#00ff88', lw=2))
        ax.text(start_x + n_cols*cell_w + 1.5, start_y + n_rows*cell_h/2,
               'mean=0\nstd=1\n(per row)', fontsize=9, va='center', color='#00ff88')
    
    elif norm_direction == 'col':
        # Arrow pointing vertically (across batch)
        for j in range(n_cols):
            x = start_x + j * cell_w + cell_w * 0.4
            ax.annotate('', xy=(x, start_y + n_rows*cell_h + 0.2), 
                       xytext=(x, start_y + n_rows*cell_h + 0.8),
                       arrowprops=dict(arrowstyle='<-', color='#ffd93d', lw=2))
        ax.text(start_x + n_cols*cell_w/2, start_y + n_rows*cell_h + 1.2,
               'mean=0, std=1 (per column)', fontsize=9, ha='center', color='#ffd93d')

draw_3d_tensor(axes[0], 'Raw Data\n(Different scales everywhere)', None, None)
draw_3d_tensor(axes[1], 'BatchNorm\n(Normalize across BATCH dimension)', 'col', '#ffd93d')
draw_3d_tensor(axes[2], 'LayerNorm / RMSNorm\n(Normalize across FEATURE dimension)', 'row', '#00ff88')

# Add explanation text
explanation = """
Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• BatchNorm:  Normalize each FEATURE (column) across all samples  →  Each column has mean=0, std=1
• LayerNorm:  Normalize each SAMPLE (row) across all features     →  Each row has mean=0, std=1  
• RMSNorm:    Like LayerNorm but only SCALE (no mean centering)   →  Each row has RMS=1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why LLM uses LayerNorm/RMSNorm instead of BatchNorm?
  1. NLP has variable sequence lengths (hard to batch)
  2. BatchNorm requires large batch size for stable statistics
  3. LayerNorm is independent of batch size → works with batch_size=1
"""
fig2.text(0.5, -0.05, explanation, ha='center', va='top', fontsize=10,
         family='monospace', color='#ccc',
         bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#444', pad=0.5))

plt.tight_layout(rect=[0, 0.15, 1, 0.95])
plt.savefig('normalization_direction.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("Saved: normalization_direction.png")

# ============================================================
# Figure 3: LayerNorm vs RMSNorm detailed comparison
# ============================================================
fig3, axes = plt.subplots(2, 2, figsize=(12, 10))

# Create a single sample with non-zero mean
np.random.seed(123)
x_single = np.random.randn(1, 32) * 2 + 3  # mean ≈ 3, std ≈ 2

# Apply normalizations
x_ln_single = layer_norm(x_single)
x_rms_single = rms_norm(x_single)

dims = np.arange(32)

# Plot 1: Raw data
ax1 = axes[0, 0]
ax1.bar(dims, x_single.flatten(), color='#e94560', alpha=0.8)
ax1.axhline(x_single.mean(), color='white', linestyle='--', label=f'mean={x_single.mean():.2f}')
ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax1.set_title(f'Raw Data\nmean={x_single.mean():.2f}, std={x_single.std():.2f}', fontsize=11, fontweight='bold')
ax1.set_xlabel('Feature Dimension')
ax1.set_ylabel('Value')
ax1.legend()
ax1.set_ylim(-3, 8)

# Plot 2: After LayerNorm
ax2 = axes[0, 1]
ax2.bar(dims, x_ln_single.flatten(), color='#00d9ff', alpha=0.8)
ax2.axhline(0, color='white', linestyle='--', label='mean=0')
ax2.set_title(f'After LayerNorm\nmean={x_ln_single.mean():.4f}, std={x_ln_single.std():.2f}', 
              fontsize=11, fontweight='bold')
ax2.set_xlabel('Feature Dimension')
ax2.set_ylabel('Value')
ax2.legend()
ax2.set_ylim(-3, 8)

# Plot 3: After RMSNorm
ax3 = axes[1, 0]
ax3.bar(dims, x_rms_single.flatten(), color='#00ff88', alpha=0.8)
ax3.axhline(x_rms_single.mean(), color='white', linestyle='--', 
            label=f'mean={x_rms_single.mean():.2f} (NOT 0!)')
ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax3.set_title(f'After RMSNorm\nmean={x_rms_single.mean():.2f} (preserved!), RMS=1', 
              fontsize=11, fontweight='bold')
ax3.set_xlabel('Feature Dimension')
ax3.set_ylabel('Value')
ax3.legend()
ax3.set_ylim(-3, 8)

# Plot 4: Side by side comparison
ax4 = axes[1, 1]
width = 0.35
ax4.bar(dims - width/2, x_ln_single.flatten(), width, label='LayerNorm', color='#00d9ff', alpha=0.8)
ax4.bar(dims + width/2, x_rms_single.flatten(), width, label='RMSNorm', color='#00ff88', alpha=0.8)
ax4.axhline(0, color='white', linestyle='--', alpha=0.5)
ax4.set_title('LayerNorm vs RMSNorm\n(Same input, different treatment of mean)', 
              fontsize=11, fontweight='bold')
ax4.set_xlabel('Feature Dimension')
ax4.set_ylabel('Normalized Value')
ax4.legend()

# Add annotation
ax4.annotate('LayerNorm subtracts mean\n→ centered at 0', 
             xy=(5, x_ln_single.flatten()[5]), xytext=(10, 2),
             fontsize=9, color='#00d9ff',
             arrowprops=dict(arrowstyle='->', color='#00d9ff'))
ax4.annotate('RMSNorm keeps relative offset\n→ NOT centered at 0', 
             xy=(25, x_rms_single.flatten()[25]), xytext=(18, -2),
             fontsize=9, color='#00ff88',
             arrowprops=dict(arrowstyle='->', color='#00ff88'))

plt.suptitle('LayerNorm vs RMSNorm: The Mean Difference', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('layernorm_vs_rmsnorm.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print("Saved: layernorm_vs_rmsnorm.png")

print("\n✅ All v2 normalization visualizations completed!")
