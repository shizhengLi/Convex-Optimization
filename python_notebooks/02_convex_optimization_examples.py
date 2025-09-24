"""
凸优化问题的求解示例
"""

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.patches import Polygon

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def linear_programming_example():
    """线性规划示例：资源分配问题"""

    print("=== 线性规划示例 ===")

    # 问题：最大化利润
    # max 3x + 4y
    # s.t. x + 2y <= 8
    #      3x + y <= 9
    #      x >= 0, y >= 0

    # 创建变量
    x = cp.Variable()
    y = cp.Variable()

    # 创建目标函数
    objective = cp.Maximize(3*x + 4*y)

    # 创建约束
    constraints = [
        x + 2*y <= 8,
        3*x + y <= 9,
        x >= 0,
        y >= 0
    ]

    # 创建问题并求解
    problem = cp.Problem(objective, constraints)
    problem.solve()

    print(f"最优解: x = {x.value:.2f}, y = {y.value:.2f}")
    print(f"最优值: {problem.value:.2f}")
    print(f"求解状态: {problem.status}")

    # 可视化
    visualize_linear_programming(x.value, y.value)

def visualize_linear_programming(opt_x, opt_y):
    """可视化线性规划问题"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 绘制约束条件
    x = np.linspace(0, 4, 100)

    # 约束1: x + 2y <= 8
    y1 = (8 - x) / 2
    ax.plot(x, y1, 'b-', linewidth=2, label='$x + 2y \\leq 8$')

    # 约束2: 3x + y <= 9
    y2 = 9 - 3*x
    ax.plot(x, y2, 'r-', linewidth=2, label='$3x + y \\leq 9$')

    # 非负约束
    ax.axhline(y=0, color='k', linewidth=1, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=1, alpha=0.3)

    # 绘制可行域
    x_feasible = np.array([0, 0, 2, 3, 0])
    y_feasible = np.array([0, 4, 3, 0, 0])
    feasible_region = Polygon(list(zip(x_feasible, y_feasible)),
                            alpha=0.3, facecolor='lightgreen', label='可行域')
    ax.add_patch(feasible_region)

    # 绘制目标函数的等高线
    x_obj = np.linspace(0, 4, 20)
    y_obj = np.linspace(0, 4, 20)
    X, Y = np.meshgrid(x_obj, y_obj)
    Z = 3*X + 4*Y

    contours = ax.contour(X, Y, Z, levels=10, colors='gray', alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=8)

    # 标记最优解
    ax.plot(opt_x, opt_y, 'ro', markersize=10, label=f'最优解 ({opt_x:.1f}, {opt_y:.1f})')

    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('线性规划：最大化 $3x + 4y$')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('../images/linear_programming_example.png', dpi=300, bbox_inches='tight')
    plt.show()

def quadratic_programming_example():
    """二次规划示例：最小化二次函数"""

    print("\n=== 二次规划示例 ===")

    # 问题：最小化二次函数
    # min 0.5 * x^T * P * x + q^T * x
    # s.t. Ax <= b

    # 创建变量
    x = cp.Variable(2)

    # 定义二次目标函数
    P = np.array([[2, 1], [1, 2]])  # 正定矩阵
    q = np.array([-1, -1])

    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)

    # 添加约束
    A = np.array([[1, 1], [-1, 0], [0, -1]])
    b = np.array([1, 0, 0])

    constraints = [A @ x <= b]

    # 求解
    problem = cp.Problem(objective, constraints)
    problem.solve()

    print(f"最优解: x = [{x.value[0]:.3f}, {x.value[1]:.3f}]")
    print(f"最优值: {problem.value:.3f}")
    print(f"求解状态: {problem.status}")

    # 可视化
    visualize_quadratic_programming(x.value, P, q)

def visualize_quadratic_programming(opt_x, P, q):
    """可视化二次规划问题"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 绘制目标函数的等高线
    x = np.linspace(-0.5, 1.5, 100)
    y = np.linspace(-0.5, 1.5, 100)
    X, Y = np.meshgrid(x, y)

    # 计算目标函数值
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i,j], Y[i,j]])
            Z[i,j] = 0.5 * point.T @ P @ point + q.T @ point

    contours = ax.contour(X, Y, Z, levels=20, colors='gray', alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=8)

    # 绘制约束
    x_line = np.linspace(-0.5, 1.5, 100)

    # 约束 x1 + x2 <= 1
    y_line = 1 - x_line
    ax.plot(x_line, y_line, 'b-', linewidth=2, label='$x_1 + x_2 \\leq 1$')

    # 非负约束
    ax.axhline(y=0, color='r', linewidth=2, label='$x_2 \\geq 0$')
    ax.axvline(x=0, color='g', linewidth=2, label='$x_1 \\geq 0$')

    # 填充可行域
    x_feas = np.array([0, 1, 0, 0])
    y_feas = np.array([0, 0, 1, 0])
    feasible_region = Polygon(list(zip(x_feas, y_feas)),
                            alpha=0.3, facecolor='lightgreen', label='可行域')
    ax.add_patch(feasible_region)

    # 标记最优解
    ax.plot(opt_x[0], opt_x[1], 'ro', markersize=10,
            label=f'最优解 ({opt_x[0]:.2f}, {opt_x[1]:.2f})')

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title('二次规划：最小化 $\\frac{1}{2}x^TPx + q^Tx$')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('../images/quadratic_programming_example.png', dpi=300, bbox_inches='tight')
    plt.show()

def l1_regression_example():
    """L1回归示例：鲁棒回归"""

    print("\n=== L1回归示例 ===")

    # 生成带异常值的数据
    np.random.seed(42)
    n_samples = 20
    x = np.linspace(0, 10, n_samples)
    y_true = 2 * x + 1 + np.random.normal(0, 1, n_samples)

    # 添加异常值
    y_true[5] += 10  # 异常值
    y_true[15] -= 8  # 异常值

    # 使用CVXPY求解L1回归
    # min ||Ax - b||_1
    A = np.vstack([x, np.ones(n_samples)]).T
    b = y_true

    beta = cp.Variable(2)
    objective = cp.Minimize(cp.norm(A @ beta - b, 1))
    problem = cp.Problem(objective)
    problem.solve()

    # 对比普通最小二乘法
    beta_ls = np.linalg.lstsq(A, b, rcond=None)[0]

    print(f"L1回归解: 斜率={beta.value[0]:.3f}, 截距={beta.value[1]:.3f}")
    print(f"最小二乘解: 斜率={beta_ls[0]:.3f}, 截距={beta_ls[1]:.3f}")

    # 可视化
    visualize_l1_regression(x, y_true, beta.value, beta_ls)

def visualize_l1_regression(x, y, beta_l1, beta_ls):
    """可视化L1回归和最小二乘回归的对比"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 绘制数据点
    ax.scatter(x, y, c='red', s=50, alpha=0.7, label='数据点')

    # 绘制L1回归线
    x_line = np.linspace(0, 10, 100)
    y_l1 = beta_l1[0] * x_line + beta_l1[1]
    ax.plot(x_line, y_l1, 'b-', linewidth=3, label='L1回归（鲁棒）')

    # 绘制最小二乘回归线
    y_ls = beta_ls[0] * x_line + beta_ls[1]
    ax.plot(x_line, y_ls, 'g--', linewidth=3, label='最小二乘回归')

    # 标记异常值
    outliers = [5, 15]
    ax.scatter(x[outliers], y[outliers], c='orange', s=100,
               marker='x', linewidth=3, label='异常值')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('L1回归 vs 最小二乘回归（鲁棒性对比）')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('../images/l1_regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def portfolio_optimization_example():
    """投资组合优化示例"""

    print("\n=== 投资组合优化示例 ===")

    # 生成资产的期望收益和协方差矩阵
    np.random.seed(42)
    n_assets = 4

    # 期望收益率
    mu = np.array([0.08, 0.12, 0.15, 0.10])

    # 协方差矩阵
    Sigma = np.array([
        [0.04, 0.02, 0.01, 0.015],
        [0.02, 0.09, 0.03, 0.02],
        [0.01, 0.03, 0.16, 0.025],
        [0.015, 0.02, 0.025, 0.0625]
    ])

    # 求解最小方差投资组合
    w = cp.Variable(n_assets)

    # 目标：最小化风险（方差）
    risk = cp.quad_form(w, Sigma)
    objective = cp.Minimize(risk)

    # 约束：权重和为1，期望收益率不低于目标
    target_return = 0.12
    constraints = [
        cp.sum(w) == 1,
        mu @ w >= target_return,
        w >= 0  # 不允许做空
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    print(f"最优投资组合权重: {w.value}")
    print(f"最小风险: {np.sqrt(problem.value):.4f}")
    print(f"期望收益率: {mu @ w.value:.4f}")

    # 可视化有效前沿
    visualize_efficient_frontier(mu, Sigma)

def visualize_efficient_frontier(mu, Sigma):
    """可视化投资组合的有效前沿"""

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # 计算有效前沿上的点
    target_returns = np.linspace(0.08, 0.15, 20)
    risks = []

    for target_return in target_returns:
        w = cp.Variable(len(mu))
        risk = cp.quad_form(w, Sigma)
        objective = cp.Minimize(risk)
        constraints = [
            cp.sum(w) == 1,
            mu @ w >= target_return,
            w >= 0
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        risks.append(np.sqrt(problem.value))

    # 绘制有效前沿
    ax.plot(risks, target_returns, 'b-', linewidth=3, label='有效前沿')

    # 标记各个资产
    asset_risks = np.sqrt(np.diag(Sigma))
    ax.scatter(asset_risks, mu, c='red', s=100, alpha=0.7, label='单个资产')

    # 标记最小方差投资组合
    ax.scatter(risks[0], target_returns[0], c='green', s=150,
               marker='*', label='最小方差组合')

    ax.set_xlabel('风险（标准差）')
    ax.set_ylabel('期望收益率')
    ax.set_title('投资组合有效前沿')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('../images/portfolio_efficient_frontier.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("生成凸优化问题求解示例...")

    # 运行所有示例
    linear_programming_example()
    quadratic_programming_example()
    l1_regression_example()
    portfolio_optimization_example()

    print("\n所有示例完成！图形已保存到 ../images/ 目录")