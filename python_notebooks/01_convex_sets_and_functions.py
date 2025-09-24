"""
凸集合和凸函数的可视化示例
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.patches as patches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_convex_vs_nonconvex_sets():
    """绘制凸集和非凸集的对比"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 凸集示例：圆
    circle = Circle((0, 0), 1, fill=False, color='blue', linewidth=2)
    ax1.add_patch(circle)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('凸集示例：圆形')

    # 标记两点和连线
    x1, y1 = 0.5, 0.5
    x2, y2 = -0.7, -0.7
    ax1.plot([x1, x2], [y1, y2], 'r--', linewidth=2, alpha=0.7)
    ax1.plot(x1, y1, 'ro', markersize=8)
    ax1.plot(x2, y2, 'ro', markersize=8)

    # 非凸集示例：月牙形
    theta = np.linspace(0, 2*np.pi, 100)
    x_outer = 1.5 * np.cos(theta)
    y_outer = 1.5 * np.sin(theta)
    x_inner = 0.8 * np.cos(theta) + 0.3
    y_inner = 0.8 * np.sin(theta)

    ax2.fill(x_outer, y_outer, 'lightblue', alpha=0.7, label='外圆')
    ax2.fill(x_inner, y_inner, 'white', alpha=1, label='内圆')
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('非凸集示例：月牙形')

    # 标记两点和连线（部分在集合外）
    x1, y1 = 1.2, 0
    x2, y2 = -1.2, 0
    ax2.plot([x1, x2], [y1, y2], 'r--', linewidth=2, alpha=0.7)
    ax2.plot(x1, y1, 'ro', markersize=8)
    ax2.plot(x2, y2, 'ro', markersize=8)

    plt.tight_layout()
    plt.savefig('../images/convex_vs_nonconvex_sets.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_convex_functions():
    """绘制凸函数和非凸函数的对比"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    x = np.linspace(-3, 3, 100)

    # 凸函数1：二次函数
    y1 = x**2
    ax1.plot(x, y1, 'b-', linewidth=2, label='$f(x) = x^2$')
    ax1.set_title('严格凸函数：$f(x) = x^2$')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 标记两点和割线
    x_points = [-1, 1]
    y_points = [1, 1]
    ax1.plot(x_points, y_points, 'ro', markersize=8)
    ax1.plot(x_points, y_points, 'r--', linewidth=2, alpha=0.7, label='割线')

    # 凸函数2：指数函数
    y2 = np.exp(x)
    ax2.plot(x, y2, 'g-', linewidth=2, label='$f(x) = e^x$')
    ax2.set_title('严格凸函数：$f(x) = e^x$')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 非凸函数1：三次函数
    y3 = x**3 - 3*x
    ax3.plot(x, y3, 'r-', linewidth=2, label='$f(x) = x^3 - 3x$')
    ax3.set_title('非凸函数：$f(x) = x^3 - 3x$')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # 凸函数3：绝对值函数
    y4 = np.abs(x)
    ax4.plot(x, y4, 'm-', linewidth=2, label='$f(x) = |x|$')
    ax4.set_title('凸函数（不可导）：$f(x) = |x|$')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('../images/convex_functions_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_convex_hull():
    """绘制凸包示例"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # 生成一些随机点
    np.random.seed(42)
    points = np.random.randn(20, 2)

    # 绘制所有点
    ax.scatter(points[:, 0], points[:, 1], c='red', s=50, alpha=0.7, label='原始点')

    # 计算并绘制凸包
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)

    # 绘制凸包
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=2)

    # 填充凸包
    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
            'lightblue', alpha=0.3, label='凸包')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('凸包示例')
    ax.legend()

    plt.tight_layout()
    plt.savefig('../images/convex_hull_example.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_jensen_inequality():
    """绘制Jensen不等式的可视化"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x = np.linspace(-2, 2, 100)
    f = x**2  # 凸函数

    ax.plot(x, f, 'b-', linewidth=3, label='$f(x) = x^2$')

    # 选择两点
    x1, x2 = -1, 1
    y1, y2 = f[np.where(x == x1)[0][0]], f[np.where(x == x2)[0][0]]

    # 绘制两点
    ax.plot([x1, x2], [y1, y2], 'ro', markersize=10, label='端点')

    # 绘制割线
    theta_values = np.linspace(0, 1, 100)
    line_x = theta_values * x1 + (1 - theta_values) * x2
    line_y = theta_values * y1 + (1 - theta_values) * y2
    ax.plot(line_x, line_y, 'r--', linewidth=2, label='割线')

    # 选择特定的theta值
    theta = 0.3
    x_theta = theta * x1 + (1 - theta) * x2
    f_x_theta = x_theta**2
    line_x_theta = theta * y1 + (1 - theta) * y2

    # 标记Jensen不等式
    ax.plot([x_theta, x_theta], [f_x_theta, line_x_theta], 'g-', linewidth=3,
            label=f'$f(\\theta x_1 + (1-\\theta)x_2) \\leq \\theta f(x_1) + (1-\\theta)f(x_2)$')
    ax.plot(x_theta, f_x_theta, 'go', markersize=8)
    ax.plot(x_theta, line_x_theta, 'ro', markersize=8)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 4)
    ax.grid(True, alpha=0.3)
    ax.set_title('Jensen不等式可视化：$f(\\theta x_1 + (1-\\theta)x_2) \\leq \\theta f(x_1) + (1-\\theta)f(x_2)$')
    ax.legend()

    plt.tight_layout()
    plt.savefig('../images/jensen_inequality.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("生成凸集合和凸函数的可视化图形...")

    # 生成所有图形
    plot_convex_vs_nonconvex_sets()
    plot_convex_functions()
    plot_convex_hull()
    plot_jensen_inequality()

    print("所有图形已保存到 ../images/ 目录")