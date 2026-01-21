"""
Interactive Optimizer Visualization with Ball Animation
Compare: SGD, Momentum, AdaGrad, RMSprop, Adam

Usage:
  python optimizer_animation.py           # Generate GIF animation
  python optimizer_animation.py --interactive  # Interactive mode (requires display)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.gridspec as gridspec
import sys

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = '#1a1a2e'
plt.rcParams['axes.facecolor'] = '#16213e'
plt.rcParams['axes.labelcolor'] = '#eee'
plt.rcParams['text.color'] = '#eee'
plt.rcParams['xtick.color'] = '#eee'
plt.rcParams['ytick.color'] = '#eee'

# ============================================================
# Loss Function: Beale function (challenging optimization landscape)
# ============================================================


def beale(x, y):
    """Beale function - has a global minimum at (3, 0.5)"""
    return ((1.5 - x + x * y)**2 +
            (2.25 - x + x * y**2)**2 +
            (2.625 - x + x * y**3)**2)


def beale_grad(x, y):
    """Gradient of Beale function"""
    dx = (2 * (1.5 - x + x * y) * (-1 + y) +
          2 * (2.25 - x + x * y**2) * (-1 + y**2) +
          2 * (2.625 - x + x * y**3) * (-1 + y**3))
    dy = (2 * (1.5 - x + x * y) * x +
          2 * (2.25 - x + x * y**2) * (2 * x * y) +
          2 * (2.625 - x + x * y**3) * (3 * x * y**2))
    return np.array([dx, dy])

# Simpler function for clearer visualization


def quadratic(x, y):
    """Elongated quadratic - shows momentum benefits clearly"""
    return 0.1 * x**2 + 2 * y**2


def quadratic_grad(x, y):
    return np.array([0.2 * x, 4 * y])

# Rosenbrock function


def rosenbrock(x, y):
    """Rosenbrock function - classic optimization test"""
    return (1 - x)**2 + 100 * (y - x**2)**2


def rosenbrock_grad(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# ============================================================
# Optimizer Classes
# ============================================================


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.name = f"SGD (lr={lr})"

    def step(self, pos, grad):
        return pos - self.lr * grad


class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = np.zeros(2)
        self.name = f"Momentum (lr={lr}, β={beta})"

    def step(self, pos, grad):
        self.v = self.beta * self.v + grad
        return pos - self.lr * self.v


class AdaGrad:
    def __init__(self, lr=0.5, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.G = np.zeros(2)
        self.name = f"AdaGrad (lr={lr})"

    def step(self, pos, grad):
        self.G += grad ** 2
        return pos - self.lr * grad / (np.sqrt(self.G) + self.eps)


class RMSprop:
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.v = np.zeros(2)
        self.name = f"RMSprop (lr={lr}, β={beta})"

    def step(self, pos, grad):
        self.v = self.beta * self.v + (1 - self.beta) * grad ** 2
        return pos - self.lr * grad / (np.sqrt(self.v) + self.eps)


class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(2)
        self.v = np.zeros(2)
        self.t = 0
        self.name = f"Adam (lr={lr}, β1={beta1}, β2={beta2})"

    def step(self, pos, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return pos - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# ============================================================
# Run optimization and collect trajectory
# ============================================================


def optimize(optimizer, loss_func, grad_func, start_pos, n_steps=100):
    trajectory = [start_pos.copy()]
    pos = start_pos.copy()

    for _ in range(n_steps):
        grad = grad_func(pos[0], pos[1])
        # Clip gradient to prevent explosion
        grad = np.clip(grad, -10, 10)
        pos = optimizer.step(pos, grad)
        # Clip position to stay in visible area
        pos = np.clip(pos, -4.5, 4.5)
        trajectory.append(pos.copy())

    return np.array(trajectory)

# ============================================================
# Generate Animation GIF
# ============================================================


def create_animation():
    print("Creating optimizer comparison animation...")

    # Setup
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.25)

    # Loss surface plot
    ax_main = fig.add_subplot(gs[0, :2])

    # Create meshgrid for contour
    x = np.linspace(-4.5, 4.5, 200)
    y = np.linspace(-4.5, 4.5, 200)
    X, Y = np.meshgrid(x, y)

    # Use quadratic function (clearer visualization)
    Z = quadratic(X, Y)
    loss_func = quadratic
    grad_func = quadratic_grad
    start_pos = np.array([-4.0, 3.0])
    n_steps = 80

    # Plot contour
    levels = np.logspace(-1, 2, 30)
    ax_main.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax_main.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax_main.set_xlabel('x', fontsize=12)
    ax_main.set_ylabel('y', fontsize=12)
    ax_main.set_title('Optimizer Trajectories on Loss Surface\n$L(x,y) = 0.1x^2 + 2y^2$ (elongated valley)',
                      fontsize=14, fontweight='bold')

    # Mark minimum
    ax_main.plot(0, 0, 'r*', markersize=20, label='Global Minimum')

    # Create optimizers with different settings
    optimizers = [
        SGD(lr=0.1),
        Momentum(lr=0.1, beta=0.9),
        AdaGrad(lr=1.0),
        RMSprop(lr=0.1, beta=0.9),
        Adam(lr=0.3, beta1=0.9, beta2=0.999),
    ]

    colors = ['#e94560', '#ffd93d', '#00d9ff', '#00ff88', '#9b59b6']

    # Run all optimizations
    trajectories = []
    for opt in optimizers:
        traj = optimize(opt, loss_func, grad_func, start_pos.copy(), n_steps)
        trajectories.append(traj)

    # Initialize plot elements
    lines = []
    balls = []
    for i, opt in enumerate(optimizers):
        line, = ax_main.plot([], [], '-', color=colors[i], linewidth=2, alpha=0.7, label=opt.name)
        ball, = ax_main.plot([], [], 'o', color=colors[i], markersize=12, markeredgecolor='white', markeredgewidth=2)
        lines.append(line)
        balls.append(ball)

    ax_main.legend(loc='upper right', fontsize=9)

    # Loss curve subplot
    ax_loss = fig.add_subplot(gs[0, 2])
    ax_loss.set_xlabel('Step', fontsize=11)
    ax_loss.set_ylabel('Loss (log scale)', fontsize=11)
    ax_loss.set_title('Loss vs Steps', fontsize=12, fontweight='bold')
    ax_loss.set_yscale('log')
    ax_loss.set_xlim(0, n_steps)

    loss_lines = []
    for i, opt in enumerate(optimizers):
        line, = ax_loss.plot([], [], '-', color=colors[i], linewidth=2, label=opt.name.split()[0])
        loss_lines.append(line)
    ax_loss.legend(fontsize=8)

    # Info panels
    ax_info = fig.add_subplot(gs[1, :])
    ax_info.axis('off')

    info_text = """
    Optimizer Comparison Summary:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • SGD:       Basic gradient descent. Oscillates in narrow valleys.
    • Momentum:  Adds velocity term. Dampens oscillations, accelerates in consistent directions.
    • AdaGrad:   Adapts learning rate per-parameter. Good for sparse gradients, but LR decays too fast.
    • RMSprop:   Like AdaGrad but uses moving average. Prevents LR from decaying to zero.
    • Adam:      Combines Momentum (1st moment) + RMSprop (2nd moment). The default choice for most tasks.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    ax_info.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10,
                 family='monospace', transform=ax_info.transAxes,
                 bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#00d9ff', pad=0.5))

    # Animation function
    def animate(frame):
        for i, (traj, line, ball, loss_line) in enumerate(zip(trajectories, lines, balls, loss_lines)):
            # Update trajectory line
            line.set_data(traj[:frame + 1, 0], traj[:frame + 1, 1])
            # Update ball position
            ball.set_data([traj[frame, 0]], [traj[frame, 1]])
            # Update loss curve
            losses = [loss_func(traj[j, 0], traj[j, 1]) for j in range(frame + 1)]
            loss_line.set_data(range(frame + 1), losses)

        # Update y-axis limits for loss plot
        all_losses = []
        for traj in trajectories:
            all_losses.extend([loss_func(traj[j, 0], traj[j, 1]) for j in range(frame + 1)])
        if all_losses:
            ax_loss.set_ylim(max(min(all_losses) * 0.5, 1e-4), max(all_losses) * 2)

        return lines + balls + loss_lines

    # Create animation
    anim = FuncAnimation(fig, animate, frames=n_steps, interval=100, blit=True)

    # Save as GIF
    print("Saving animation as GIF (this may take a minute)...")
    writer = PillowWriter(fps=10)
    anim.save('optimizer_animation.gif', writer=writer, dpi=100,
              savefig_kwargs={'facecolor': '#1a1a2e'})
    plt.close()
    print("Saved: optimizer_animation.gif")

# ============================================================
# Interactive Mode with Sliders
# ============================================================


def create_interactive():
    print("Creating interactive optimizer visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, height_ratios=[3, 0.5, 0.5], hspace=0.4, wspace=0.3)

    # Main plot
    ax_main = fig.add_subplot(gs[0, :2])

    # Create meshgrid
    x = np.linspace(-4.5, 4.5, 100)
    y = np.linspace(-4.5, 4.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = quadratic(X, Y)

    # Plot contour
    levels = np.logspace(-1, 2, 20)
    ax_main.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
    ax_main.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.3)
    ax_main.plot(0, 0, 'r*', markersize=20)
    ax_main.set_title('Optimizer Comparison (Adjust sliders below)', fontsize=14, fontweight='bold')
    ax_main.set_xlabel('x')
    ax_main.set_ylabel('y')

    # Loss plot
    ax_loss = fig.add_subplot(gs[0, 2])
    ax_loss.set_xlabel('Step')
    ax_loss.set_ylabel('Loss (log scale)')
    ax_loss.set_title('Loss Curve', fontsize=12, fontweight='bold')
    ax_loss.set_yscale('log')

    # Slider axes
    ax_lr = fig.add_axes([0.15, 0.25, 0.3, 0.03])
    ax_beta1 = fig.add_axes([0.15, 0.2, 0.3, 0.03])
    ax_beta2 = fig.add_axes([0.15, 0.15, 0.3, 0.03])
    ax_steps = fig.add_axes([0.55, 0.25, 0.3, 0.03])

    # Create sliders
    s_lr = Slider(ax_lr, 'Learning Rate', 0.01, 1.0, valinit=0.2, color='#00d9ff')
    s_beta1 = Slider(ax_beta1, 'Beta1 (Momentum)', 0.0, 0.99, valinit=0.9, color='#00ff88')
    s_beta2 = Slider(ax_beta2, 'Beta2 (RMSprop)', 0.9, 0.999, valinit=0.999, color='#ffd93d')
    s_steps = Slider(ax_steps, 'Steps', 10, 200, valinit=80, valstep=10, color='#9b59b6')

    # Button axis
    ax_button = fig.add_axes([0.4, 0.08, 0.2, 0.05])
    button = Button(ax_button, 'Run Optimization', color='#16213e', hovercolor='#e94560')

    colors = ['#e94560', '#ffd93d', '#00d9ff', '#00ff88', '#9b59b6']
    opt_names = ['SGD', 'Momentum', 'AdaGrad', 'RMSprop', 'Adam']

    # Store plot elements
    plot_elements = {'lines': [], 'loss_lines': []}

    def update(event=None):
        # Clear previous plots
        for line in plot_elements['lines']:
            line.remove()
        for line in plot_elements['loss_lines']:
            line.remove()
        plot_elements['lines'] = []
        plot_elements['loss_lines'] = []

        # Get parameters
        lr = s_lr.val
        beta1 = s_beta1.val
        beta2 = s_beta2.val
        n_steps = int(s_steps.val)

        # Create optimizers
        optimizers = [
            SGD(lr=lr),
            Momentum(lr=lr, beta=beta1),
            AdaGrad(lr=lr * 5),  # AdaGrad needs higher initial LR
            RMSprop(lr=lr, beta=beta2),
            Adam(lr=lr, beta1=beta1, beta2=beta2),
        ]

        start_pos = np.array([-4.0, 3.0])

        ax_loss.clear()
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss (log scale)')
        ax_loss.set_title('Loss Curve', fontsize=12, fontweight='bold')
        ax_loss.set_yscale('log')

        for i, (opt, name) in enumerate(zip(optimizers, opt_names)):
            traj = optimize(opt, quadratic, quadratic_grad, start_pos.copy(), n_steps)

            # Plot trajectory
            line, = ax_main.plot(traj[:, 0], traj[:, 1], '-o', color=colors[i],
                                 linewidth=2, markersize=3, alpha=0.7, label=name)
            # Plot final position
            ax_main.plot(traj[-1, 0], traj[-1, 1], 's', color=colors[i],
                         markersize=12, markeredgecolor='white', markeredgewidth=2)
            plot_elements['lines'].append(line)

            # Plot loss curve
            losses = [quadratic(traj[j, 0], traj[j, 1]) for j in range(len(traj))]
            loss_line, = ax_loss.plot(losses, color=colors[i], linewidth=2, label=name)
            plot_elements['loss_lines'].append(loss_line)

        ax_main.legend(loc='upper right', fontsize=9)
        ax_loss.legend(loc='upper right', fontsize=8)
        fig.canvas.draw_idle()

    button.on_clicked(update)

    # Initial plot
    update()

    plt.savefig('optimizer_interactive.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    print("Saved: optimizer_interactive.png")

    # Show if display available
    try:
        plt.show()
    except BaseException:
        print("(No display available, saved static image)")

# ============================================================
# Create static comparison with different parameter settings
# ============================================================


def create_parameter_comparison():
    print("Creating parameter comparison plots...")

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.25)

    # Setup
    x = np.linspace(-4.5, 4.5, 100)
    y = np.linspace(-4.5, 4.5, 100)
    X, Y = np.meshgrid(x, y)
    Z = quadratic(X, Y)
    levels = np.logspace(-1, 2, 20)
    start_pos = np.array([-4.0, 3.0])
    n_steps = 60

    # 1. Learning Rate comparison (SGD)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
    ax1.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.2)
    ax1.plot(0, 0, 'r*', markersize=15)

    lrs = [0.05, 0.1, 0.2, 0.5]
    colors_lr = ['#e94560', '#ffd93d', '#00d9ff', '#00ff88']
    for lr, c in zip(lrs, colors_lr):
        opt = SGD(lr=lr)
        traj = optimize(opt, quadratic, quadratic_grad, start_pos.copy(), n_steps)
        ax1.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=2, label=f'lr={lr}')
        ax1.plot(traj[-1, 0], traj[-1, 1], 'o', color=c, markersize=8)
    ax1.set_title('SGD: Learning Rate Effect\n(Higher → Faster but may overshoot)',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    # 2. Momentum comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
    ax2.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.2)
    ax2.plot(0, 0, 'r*', markersize=15)

    betas = [0.0, 0.5, 0.9, 0.99]
    for beta, c in zip(betas, colors_lr):
        opt = Momentum(lr=0.1, beta=beta)
        traj = optimize(opt, quadratic, quadratic_grad, start_pos.copy(), n_steps)
        ax2.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=2, label=f'β={beta}')
        ax2.plot(traj[-1, 0], traj[-1, 1], 'o', color=c, markersize=8)
    ax2.set_title('Momentum: Beta Effect\n(Higher → More "inertia", smoother path)',
                  fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # 3. Adam beta1 comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
    ax3.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.2)
    ax3.plot(0, 0, 'r*', markersize=15)

    beta1s = [0.0, 0.5, 0.9, 0.99]
    for beta1, c in zip(beta1s, colors_lr):
        opt = Adam(lr=0.3, beta1=beta1, beta2=0.999)
        traj = optimize(opt, quadratic, quadratic_grad, start_pos.copy(), n_steps)
        ax3.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=2, label=f'β1={beta1}')
        ax3.plot(traj[-1, 0], traj[-1, 1], 'o', color=c, markersize=8)
    ax3.set_title('Adam: Beta1 Effect (Momentum)\n(Controls 1st moment / velocity)',
                  fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

    # 4. Adam beta2 comparison
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
    ax4.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.2)
    ax4.plot(0, 0, 'r*', markersize=15)

    beta2s = [0.9, 0.99, 0.999, 0.9999]
    for beta2, c in zip(beta2s, colors_lr):
        opt = Adam(lr=0.3, beta1=0.9, beta2=beta2)
        traj = optimize(opt, quadratic, quadratic_grad, start_pos.copy(), n_steps)
        ax4.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=2, label=f'β2={beta2}')
        ax4.plot(traj[-1, 0], traj[-1, 1], 'o', color=c, markersize=8)
    ax4.set_title('Adam: Beta2 Effect (RMSprop)\n(Controls 2nd moment / adaptive LR)',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')

    # 5. All optimizers comparison
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.5)
    ax5.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.2)
    ax5.plot(0, 0, 'r*', markersize=15)

    optimizers = [
        (SGD(lr=0.1), '#e94560'),
        (Momentum(lr=0.1, beta=0.9), '#ffd93d'),
        (AdaGrad(lr=1.0), '#00d9ff'),
        (RMSprop(lr=0.1, beta=0.9), '#00ff88'),
        (Adam(lr=0.3, beta1=0.9, beta2=0.999), '#9b59b6'),
    ]

    for opt, c in optimizers:
        traj = optimize(opt, quadratic, quadratic_grad, start_pos.copy(), n_steps)
        ax5.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=2, label=opt.name.split()[0])
        ax5.plot(traj[-1, 0], traj[-1, 1], 'o', color=c, markersize=8)
    ax5.set_title('All Optimizers Comparison\n(Same starting point, different paths)',
                  fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')

    # 6. Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary = """
    Parameter Guide:
    ════════════════════════════════════

    Learning Rate (lr):
    • Too small → Slow convergence
    • Too large → Overshooting, diverge
    • Typical: 0.001 ~ 0.1

    Beta1 (Momentum term):
    • 0 = No momentum (pure SGD)
    • 0.9 = Standard (most common)
    • 0.99 = Heavy momentum

    Beta2 (RMSprop term):
    • Controls adaptive LR decay
    • 0.999 = Standard (most common)
    • Higher = Slower adaptation

    ════════════════════════════════════
    LLM Training Typical Settings:
    • Adam with lr=1e-4 ~ 3e-4
    • β1=0.9, β2=0.95 (not 0.999!)
    • Weight decay: 0.1
    • Gradient clipping: 1.0
    """

    ax6.text(0.1, 0.5, summary, ha='left', va='center', fontsize=10,
             family='monospace', transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='#16213e', edgecolor='#00d9ff', pad=0.5))

    plt.suptitle('Optimizer Parameter Effects Visualization', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('optimizer_parameters.png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print("Saved: optimizer_parameters.png")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        create_interactive()
    else:
        # Generate all visualizations
        create_animation()
        create_parameter_comparison()
        print("\n✅ All optimizer visualizations completed!")
        print("Files created:")
        print("  - optimizer_animation.gif (animated comparison)")
        print("  - optimizer_parameters.png (parameter effects)")
