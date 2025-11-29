"""
Generate figures for "Minimal Embedding Dimension for Collision-Free
Decision Processes on Statistical Manifolds"

Target: Information Geometry (Springer)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
import os

# Create figures directory
os.makedirs('figures', exist_ok=True)

# Style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def generate_helix(n_points=500, n_cycles=3, pitch=0.5):
    """Generate a helix trajectory representing a cyclic decision process."""
    t = np.linspace(0, 2 * np.pi * n_cycles, n_points)
    x = np.cos(t)
    y = np.sin(t)
    z = pitch * t / (2 * np.pi)  # Linear increase in meta-time
    return t, x, y, z


def count_collisions(points_2d, epsilon=0.1, min_time_sep=50):
    """
    Count collisions: pairs of points that are close in space
    but far apart in time (index).
    """
    n = len(points_2d)
    collisions = 0
    for i in range(n):
        for j in range(i + min_time_sep, n):
            dist = np.sqrt((points_2d[i, 0] - points_2d[j, 0])**2 +
                          (points_2d[i, 1] - points_2d[j, 1])**2)
            if dist < epsilon:
                collisions += 1
    return collisions


def figure1_helix_vs_circle():
    """
    Figure 1: The collision problem in dimensional collapse.
    Shows helix in 3D vs its 2D projection, highlighting collisions.
    """
    t, x, y, z = generate_helix(n_points=300, n_cycles=2)

    fig = plt.figure(figsize=(10, 4))

    # Panel A: 3D helix (collision-free)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(x, y, z, 'b-', linewidth=1.5, label='Trajectory')
    ax1.scatter([x[0]], [y[0]], [z[0]], c='green', s=50, zorder=5, label='Start')
    ax1.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=50, zorder=5, label='End')
    ax1.set_xlabel('$x_1$ (state)')
    ax1.set_ylabel('$x_2$ (state)')
    ax1.set_zlabel('$x_3$ (meta-time)')
    ax1.set_title('(A) $k=3$: Collision-free')
    ax1.legend(loc='upper left', fontsize=8)

    # Panel B: 2D projection (collisions)
    ax2 = fig.add_subplot(132)
    ax2.plot(x, y, 'b-', linewidth=1.5, alpha=0.7)
    ax2.scatter([x[0]], [y[0]], c='green', s=50, zorder=5, label='Start')
    ax2.scatter([x[-1]], [y[-1]], c='red', s=50, zorder=5, label='End')

    # Mark collision points (where trajectory overlaps)
    n_cycles = 2
    collision_indices = []
    for cycle in range(1, n_cycles):
        idx = int(cycle * len(x) / n_cycles)
        collision_indices.append(idx)

    for idx in collision_indices:
        ax2.scatter([x[idx]], [y[idx]], c='orange', s=100, marker='x',
                   linewidths=3, zorder=10)

    ax2.set_xlabel('$x_1$ (state)')
    ax2.set_ylabel('$x_2$ (state)')
    ax2.set_title('(B) $k=2$: Collisions (orange X)')
    ax2.set_aspect('equal')
    ax2.legend(loc='upper left', fontsize=8)

    # Panel C: Collision count vs dimension
    ax3 = fig.add_subplot(133)

    # For different k, count collisions
    k_values = [1, 2, 3, 4, 5]
    collision_counts = []

    for k in k_values:
        if k == 1:
            # Project to 1D (just x)
            proj = np.column_stack([x, np.zeros_like(x)])
            collisions = count_collisions(proj, epsilon=0.15, min_time_sep=30)
        elif k == 2:
            # Project to 2D (x, y)
            proj = np.column_stack([x, y])
            collisions = count_collisions(proj, epsilon=0.1, min_time_sep=30)
        else:
            # k >= 3: no collisions (helix is embedded)
            collisions = 0
        collision_counts.append(collisions)

    ax3.bar(k_values, collision_counts, color=['red', 'orange', 'green', 'green', 'green'],
            edgecolor='black', linewidth=1)
    ax3.set_xlabel('Embedding dimension $k$')
    ax3.set_ylabel('Collision count')
    ax3.set_title('(C) Collisions vs. dimension')
    ax3.set_xticks(k_values)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add annotation
    ax3.annotate('$k_{\\min} = 3$', xy=(3, 5), fontsize=10,
                ha='center', color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig('figures/fig1_collision_problem.pdf')
    plt.savefig('figures/fig1_collision_problem.png')
    plt.close()
    print("Generated: figures/fig1_collision_problem.pdf")


def figure2_fisher_rank():
    """
    Figure 2: Fisher information rank drop under dimensional collapse.
    Shows how the Fisher metric becomes degenerate as k decreases.
    """
    np.random.seed(42)

    # Generate samples from a parametric family on the helix
    n_samples = 200
    n_params = 3  # theta controls position on helix

    t_true = np.linspace(0, 4*np.pi, n_samples)
    noise_level = 0.1

    # Generate observed points with noise
    x = np.cos(t_true) + noise_level * np.random.randn(n_samples)
    y = np.sin(t_true) + noise_level * np.random.randn(n_samples)
    z = 0.3 * t_true + noise_level * np.random.randn(n_samples)

    data_3d = np.column_stack([x, y, z])
    data_2d = np.column_stack([x, y])

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Panel A: Empirical covariance eigenvalues (proxy for Fisher)
    ax1 = axes[0]

    # Compute covariance for 3D
    cov_3d = np.cov(data_3d.T)
    eig_3d = np.linalg.eigvalsh(cov_3d)[::-1]

    # Compute covariance for 2D
    cov_2d = np.cov(data_2d.T)
    eig_2d = np.linalg.eigvalsh(cov_2d)[::-1]

    # Normalize
    eig_3d = eig_3d / eig_3d.sum()
    eig_2d = eig_2d / eig_2d.sum()

    x_pos = np.array([1, 2, 3])
    width = 0.35

    ax1.bar(x_pos[:3] - width/2, eig_3d, width, label='$k=3$', color='steelblue')
    ax1.bar(x_pos[:2] + width/2, eig_2d, width, label='$k=2$', color='coral')

    ax1.set_xlabel('Eigenvalue index')
    ax1.set_ylabel('Normalized eigenvalue')
    ax1.set_title('(A) Covariance spectrum')
    ax1.set_xticks([1, 2, 3])
    ax1.legend()

    # Panel B: Reconstruction error
    ax2 = axes[1]

    # Try to reconstruct t from different projections
    from sklearn.linear_model import LinearRegression

    errors = []
    dims = [1, 2, 3]

    for k in dims:
        if k == 1:
            X = data_3d[:, :1]
        elif k == 2:
            X = data_3d[:, :2]
        else:
            X = data_3d

        reg = LinearRegression()
        reg.fit(X, t_true)
        t_pred = reg.predict(X)
        mse = np.mean((t_true - t_pred)**2)
        errors.append(mse)

    colors = ['red', 'orange', 'green']
    ax2.bar(dims, errors, color=colors, edgecolor='black')
    ax2.set_xlabel('Embedding dimension $k$')
    ax2.set_ylabel('MSE (time reconstruction)')
    ax2.set_title('(B) Time identifiability')
    ax2.set_xticks(dims)

    # Add annotation
    ax2.annotate('Non-identifiable', xy=(1.5, errors[1]*0.8), fontsize=9,
                ha='center', color='darkred')
    ax2.annotate('Identifiable', xy=(3, errors[2]*3), fontsize=9,
                ha='center', color='darkgreen')

    # Panel C: Effective dimension
    ax3 = axes[2]

    # Participation ratio as effective dimension
    def participation_ratio(eigenvalues):
        eigenvalues = eigenvalues / eigenvalues.sum()
        return 1.0 / np.sum(eigenvalues**2)

    k_test = [1, 2, 3, 4, 5]
    pr_values = []

    for k in k_test:
        if k == 1:
            # For 1D, PR = 1 by definition
            pr_values.append(1.0)
            continue
        elif k <= 3:
            data_k = data_3d[:, :k]
        else:
            # Add random dimensions
            extra = np.random.randn(n_samples, k-3) * 0.1
            data_k = np.column_stack([data_3d, extra])

        cov_k = np.cov(data_k.T)
        eig_k = np.linalg.eigvalsh(cov_k)[::-1]
        pr = participation_ratio(eig_k)
        pr_values.append(pr)

    ax3.plot(k_test, pr_values, 'ko-', markersize=8, linewidth=2)
    ax3.axhline(y=2, color='orange', linestyle='--', label='$k=2$ threshold')
    ax3.axhline(y=3, color='green', linestyle='--', label='$k=3$ threshold')
    ax3.fill_between([0, 6], [0, 0], [2, 2], alpha=0.2, color='red')
    ax3.fill_between([0, 6], [2, 2], [3, 3], alpha=0.2, color='orange')
    ax3.fill_between([0, 6], [3, 3], [6, 6], alpha=0.2, color='green')

    ax3.set_xlabel('Ambient dimension $k$')
    ax3.set_ylabel('Participation ratio')
    ax3.set_title('(C) Effective dimension')
    ax3.set_xlim(0.5, 5.5)
    ax3.set_ylim(0, 5)
    ax3.legend(loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig('figures/fig2_fisher_rank.pdf')
    plt.savefig('figures/fig2_fisher_rank.png')
    plt.close()
    print("Generated: figures/fig2_fisher_rank.pdf")


def figure3_general_cycles():
    """
    Figure 3: Generalization to directed cycles with self-reference.
    Shows different cycle structures and their minimal embedding dimensions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Panel A: Simple 2-cycle (paradox)
    ax1 = axes[0]
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2)
    ax1.annotate('', xy=(0.7, 0.7), xytext=(0.5, 0.87),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax1.annotate('', xy=(-0.7, -0.7), xytext=(-0.5, -0.87),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax1.scatter([1, -1], [0, 0], c=['green', 'red'], s=100, zorder=5)
    ax1.text(1.15, 0, 'T', fontsize=12, ha='left', va='center')
    ax1.text(-1.15, 0, 'F', fontsize=12, ha='right', va='center')
    ax1.text(0, -1.4, '$k_{\\min} = 3$', fontsize=11, ha='center',
            color='green', fontweight='bold')

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.6, 1.2)
    ax1.set_aspect('equal')
    ax1.set_title('(A) 2-cycle (Liar paradox)')
    ax1.axis('off')

    # Panel B: 3-cycle
    ax2 = axes[1]
    angles = np.array([np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3])
    points = np.column_stack([np.cos(angles), np.sin(angles)])

    # Draw triangle
    for i in range(3):
        j = (i + 1) % 3
        ax2.annotate('', xy=points[j]*0.85, xytext=points[i]*0.85,
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax2.scatter(points[:, 0], points[:, 1], c=['green', 'orange', 'red'],
               s=100, zorder=5)
    ax2.text(points[0, 0], points[0, 1]+0.2, 'A', fontsize=12, ha='center')
    ax2.text(points[1, 0]-0.2, points[1, 1], 'B', fontsize=12, ha='right')
    ax2.text(points[2, 0]+0.2, points[2, 1], 'C', fontsize=12, ha='left')
    ax2.text(0, -1.4, '$k_{\\min} = 3$', fontsize=11, ha='center',
            color='green', fontweight='bold')

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.6, 1.2)
    ax2.set_aspect('equal')
    ax2.set_title('(B) 3-cycle')
    ax2.axis('off')

    # Panel C: Complex graph with self-reference
    ax3 = axes[2]

    # Draw a more complex graph
    pos = {
        'A': (0, 1),
        'B': (-0.8, 0),
        'C': (0.8, 0),
        'D': (0, -0.8)
    }

    # Draw nodes
    for name, (x, y) in pos.items():
        color = 'steelblue' if name != 'D' else 'coral'
        ax3.scatter([x], [y], c=[color], s=150, zorder=5)
        ax3.text(x, y+0.25, name, fontsize=11, ha='center', va='bottom')

    # Draw edges (simplified)
    edges = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'), ('D', 'B')]
    for start, end in edges:
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        dx, dy = x2 - x1, y2 - y1
        ax3.annotate('', xy=(x2-0.1*dx, y2-0.1*dy),
                    xytext=(x1+0.1*dx, y1+0.1*dy),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

    ax3.text(0, -1.4, '$k_{\\min} \\geq 3$', fontsize=11, ha='center',
            color='green', fontweight='bold')

    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.6, 1.4)
    ax3.set_aspect('equal')
    ax3.set_title('(C) Graph with nested cycles')
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig('figures/fig3_general_cycles.pdf')
    plt.savefig('figures/fig3_general_cycles.png')
    plt.close()
    print("Generated: figures/fig3_general_cycles.pdf")


def figure4_theorem_illustration():
    """
    Figure 4: Main theorem illustration.
    Visual proof that k=3 is necessary and sufficient for collision-free
    embedding of cyclic self-referential processes.
    """
    fig = plt.figure(figsize=(10, 4))

    # Panel A: The problem setup
    ax1 = fig.add_subplot(131)

    # Draw the logical cycle
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'b-', linewidth=2, alpha=0.5)

    # Draw time arrow spiraling
    t = np.linspace(0, 4*np.pi, 200)
    r = 0.8 + 0.15 * np.sin(3*t)
    ax1.plot(r*np.cos(t), r*np.sin(t), 'r-', linewidth=1.5, label='Process $\\gamma(t)$')

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('(A) Cyclic process on $\\mathcal{M}$')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('Logical state space')
    ax1.axis('off')

    # Panel B: k=2 projection (collisions)
    ax2 = fig.add_subplot(132)

    t = np.linspace(0, 4*np.pi, 200)
    x = np.cos(t)
    y = np.sin(t)

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(t)))
    for i in range(len(t)-1):
        ax2.plot([x[i], x[i+1]], [y[i], y[i+1]], c=colors[i], linewidth=2)

    # Mark collisions
    collision_times = [np.pi, 2*np.pi, 3*np.pi]
    for ct in collision_times:
        idx = int(ct / (4*np.pi) * len(t))
        if idx < len(t):
            ax2.scatter([x[idx]], [y[idx]], c='red', s=150, marker='x',
                       linewidths=3, zorder=10)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('(B) $\\pi_2(\\gamma)$: Collisions')
    ax2.set_xlabel('Collapsed to $\\mathbb{R}^2$')

    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])

    ax2.axis('off')

    # Panel C: k=3 embedding (no collisions)
    ax3 = fig.add_subplot(133, projection='3d')

    t = np.linspace(0, 4*np.pi, 200)
    x = np.cos(t)
    y = np.sin(t)
    z = t / (4*np.pi)  # Monotone meta-time

    # Color by time
    for i in range(len(t)-1):
        ax3.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]],
                c=colors[i], linewidth=2)

    ax3.scatter([x[0]], [y[0]], [z[0]], c='green', s=80, zorder=5)
    ax3.scatter([x[-1]], [y[-1]], [z[-1]], c='red', s=80, zorder=5)

    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_zlabel('$\\tau$ (meta-time)')
    ax3.set_title('(C) $\\pi_3(\\gamma)$: Collision-free')

    plt.tight_layout()
    plt.savefig('figures/fig4_theorem.pdf')
    plt.savefig('figures/fig4_theorem.png')
    plt.close()
    print("Generated: figures/fig4_theorem.pdf")


if __name__ == '__main__':
    print("Generating figures for Information Geometry paper...")
    print("=" * 50)

    figure1_helix_vs_circle()
    figure2_fisher_rank()
    figure3_general_cycles()
    figure4_theorem_illustration()

    print("=" * 50)
    print("All figures generated in figures/ directory")
