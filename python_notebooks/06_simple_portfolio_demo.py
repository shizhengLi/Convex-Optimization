"""
简化的投资组合优化演示
重点展示优化包的实际应用，避免复杂可视化问题
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SimplePortfolioDemo:
    """
    简化的投资组合优化演示
    """
    def __init__(self):
        # 投资组合设置
        self.n_assets = 6
        self.asset_names = ['科技股', '消费股', '医疗股', '国债', '公司债', '黄金']

        # 生成市场数据
        self.generate_data()

        # 投资约束
        self.min_weight = 0.05
        self.max_weight = 0.40
        self.target_return = 0.08

        # 当前组合
        self.current_weights = np.array([0.25, 0.20, 0.15, 0.20, 0.15, 0.05])

        print("=== 简化投资组合优化演示 ===")
        print(f"资产数量: {self.n_assets}")
        print("-" * 50)

    def generate_data(self):
        """生成模拟市场数据"""
        np.random.seed(42)

        # 年化收益率和波动率
        self.annual_returns = np.array([0.12, 0.10, 0.09, 0.04, 0.05, 0.06])
        annual_vols = np.array([0.20, 0.18, 0.16, 0.06, 0.08, 0.15])

        # 相关系数矩阵
        corr = np.array([
            [1.0, 0.7, 0.5, 0.1, 0.2, 0.3],
            [0.7, 1.0, 0.6, 0.1, 0.2, 0.3],
            [0.5, 0.6, 1.0, 0.1, 0.2, 0.3],
            [0.1, 0.1, 0.1, 1.0, 0.7, 0.2],
            [0.2, 0.2, 0.2, 0.7, 1.0, 0.3],
            [0.3, 0.3, 0.3, 0.2, 0.3, 1.0]
        ])

        # 协方差矩阵
        vol_matrix = np.diag(annual_vols)
        self.covariance_matrix = vol_matrix @ corr @ vol_matrix

        print("资产特征:")
        for i, name in enumerate(self.asset_names):
            vol = annual_vols[i]
            ret = self.annual_returns[i]
            print(f"  {name}: 收益{ret:.1%}, 波动率{vol:.1%}")

    def solve_with_cvxpy(self):
        """使用CVXPY求解"""
        print("\n=== CVXPY求解结果 ===")

        try:
            import cvxpy as cp
            start_time = time.time()

            # 定义变量和目标
            w = cp.Variable(self.n_assets)
            risk = cp.quad_form(w, self.covariance_matrix)
            ret = self.annual_returns @ w

            # 约束
            constraints = [
                cp.sum(w) == 1,
                w >= self.min_weight,
                w <= self.max_weight,
                ret >= self.target_return
            ]

            # 求解
            problem = cp.Problem(cp.Minimize(risk), constraints)
            problem.solve(verbose=False)

            solve_time = time.time() - start_time

            if problem.status == 'optimal':
                weights = w.value
                portfolio_return = np.dot(weights, self.annual_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
                sharpe = portfolio_return / portfolio_risk

                print(f"✓ 求解成功! 用时: {solve_time:.3f}秒")
                print(f"  预期收益率: {portfolio_return:.2%}")
                print(f"  预期波动率: {portfolio_risk:.2%}")
                print(f"  夏普比率: {sharpe:.2f}")

                return {
                    'weights': weights,
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe': sharpe,
                    'time': solve_time
                }
            else:
                print(f"✗ 求解失败: {problem.status}")
                return None

        except ImportError:
            print("✗ CVXPY未安装")
            return None
        except Exception as e:
            print(f"✗ 出错: {e}")
            return None

    def solve_with_cvxopt(self):
        """使用CVXOPT求解"""
        print("\n=== CVXOPT求解结果 ===")

        try:
            from cvxopt import matrix, solvers
            start_time = time.time()

            # 设置问题
            n = self.n_assets
            P = matrix(self.covariance_matrix)
            q = matrix(np.zeros(n))

            # 约束
            G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
            h = matrix(np.concatenate([-self.min_weight * np.ones(n),
                                      self.max_weight * np.ones(n)]))

            A = matrix(np.vstack([np.ones(n), self.annual_returns]))
            b = matrix(np.array([1.0, self.target_return]))

            # 求解
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G, h, A, b)

            solve_time = time.time() - start_time

            if solution['status'] == 'optimal':
                weights = np.array(solution['x']).flatten()
                portfolio_return = np.dot(weights, self.annual_returns)
                portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix, weights)))
                sharpe = portfolio_return / portfolio_risk

                print(f"✓ 求解成功! 用时: {solve_time:.3f}秒")
                print(f"  预期收益率: {portfolio_return:.2%}")
                print(f"  预期波动率: {portfolio_risk:.2%}")
                print(f"  夏普比率: {sharpe:.2f}")

                return {
                    'weights': weights,
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe': sharpe,
                    'time': solve_time
                }
            else:
                print(f"✗ 求解失败: {solution['status']}")
                return None

        except ImportError:
            print("✗ CVXOPT未安装")
            return None
        except Exception as e:
            print(f"✗ 出错: {e}")
            return None

    def calculate_efficient_frontier(self):
        """计算有效前沿"""
        print("\n=== 计算有效前沿 ===")

        try:
            import cvxpy as cp

            target_returns = np.linspace(0.05, 0.12, 20)
            risks = []
            returns = []

            for target in target_returns:
                w = cp.Variable(self.n_assets)
                risk = cp.quad_form(w, self.covariance_matrix)
                ret = self.annual_returns @ w

                constraints = [
                    cp.sum(w) == 1,
                    w >= self.min_weight,
                    w <= self.max_weight,
                    ret >= target
                ]

                problem = cp.Problem(cp.Minimize(risk), constraints)
                problem.solve(verbose=False)

                if problem.status == 'optimal':
                    risks.append(np.sqrt(problem.value))
                    returns.append(target)

            return np.array(risks), np.array(returns)

        except ImportError:
            print("CVXPY未安装，无法计算有效前沿")
            return None, None

    def plot_results(self, cvxpy_result, cvxopt_result, frontier_data):
        """绘制结果图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 有效前沿
        if frontier_data is not None:
            risks, returns = frontier_data
            ax1.plot(risks, returns, 'b-', linewidth=2, label='有效前沿')

        # 标记各资产
        asset_risks = np.sqrt(np.diag(self.covariance_matrix))
        ax1.scatter(asset_risks, self.annual_returns, c='gray', s=50, alpha=0.7, label='资产')

        # 当前组合
        current_risk = np.sqrt(np.dot(self.current_weights,
                                    np.dot(self.covariance_matrix, self.current_weights)))
        current_return = np.dot(self.current_weights, self.annual_returns)
        ax1.scatter(current_risk, current_return, c='red', s=100, marker='s', label='当前')

        # 优化结果
        if cvxpy_result:
            ax1.scatter(cvxpy_result['risk'], cvxpy_result['return'], c='green',
                       s=100, marker='^', label='CVXPY')

        if cvxopt_result:
            ax1.scatter(cvxopt_result['risk'], cvxopt_result['return'], c='blue',
                       s=100, marker='^', label='CVXOPT')

        ax1.set_xlabel('风险')
        ax1.set_ylabel('收益')
        ax1.set_title('有效前沿')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 权重对比
        x = np.arange(self.n_assets)
        width = 0.25

        ax2.bar(x - width, self.current_weights, width, label='当前', alpha=0.7)
        if cvxpy_result:
            ax2.bar(x, cvxpy_result['weights'], width, label='CVXPY', alpha=0.7)
        if cvxopt_result:
            ax2.bar(x + width, cvxopt_result['weights'], width, label='CVXOPT', alpha=0.7)

        ax2.set_xlabel('资产')
        ax2.set_ylabel('权重')
        ax2.set_title('权重对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name[:2] for name in self.asset_names])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 性能对比
        if cvxpy_result or cvxopt_result:
            solvers = []
            times = []
            sharpes = []

            if cvxpy_result:
                solvers.append('CVXPY')
                times.append(cvxpy_result['time'])
                sharpes.append(cvxpy_result['sharpe'])

            if cvxopt_result:
                solvers.append('CVXOPT')
                times.append(cvxopt_result['time'])
                sharpes.append(cvxopt_result['sharpe'])

            # 求解时间对比
            ax3.bar(solvers, times, alpha=0.7)
            ax3.set_ylabel('求解时间(秒)')
            ax3.set_title('求解性能对比')
            ax3.grid(True, alpha=0.3)

            # 夏普比率对比
            ax4.bar(solvers, sharpes, alpha=0.7)
            ax4.set_ylabel('夏普比率')
            ax4.set_title('优化质量对比')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('images/simple_portfolio_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_demo(self):
        """运行演示"""
        print("开始简化投资组合优化演示...")
        print("=" * 50)

        # 分析当前组合
        current_risk = np.sqrt(np.dot(self.current_weights,
                                    np.dot(self.covariance_matrix, self.current_weights)))
        current_return = np.dot(self.current_weights, self.annual_returns)
        current_sharpe = current_return / current_risk

        print(f"当前组合:")
        print(f"  收益率: {current_return:.2%}")
        print(f"  波动率: {current_risk:.2%}")
        print(f"  夏普比率: {current_sharpe:.2f}")

        # 使用不同优化包求解
        cvxpy_result = self.solve_with_cvxpy()
        cvxopt_result = self.solve_with_cvxopt()

        # 计算有效前沿
        frontier_data = self.calculate_efficient_frontier()

        # 绘制结果
        self.plot_results(cvxpy_result, cvxopt_result, frontier_data)

        # 生成报告
        self.generate_report(cvxpy_result, cvxopt_result, current_sharpe)

        return cvxpy_result, cvxopt_result

    def generate_report(self, cvxpy_result, cvxopt_result, current_sharpe):
        """生成报告"""
        print("\n" + "=" * 50)
        print("=== 优化结果报告 ===")
        print("=" * 50)

        print("📊 当前投资组合:")
        print(f"  • 夏普比率: {current_sharpe:.2f}")

        if cvxpy_result:
            improvement = (cvxpy_result['sharpe'] - current_sharpe) / current_sharpe
            print(f"\n💻 CVXPY优化结果:")
            print(f"  • 夏普比率: {cvxpy_result['sharpe']:.2f} (改进{improvement:+.1%})")
            print(f"  • 求解时间: {cvxpy_result['time']:.3f}秒")
            print(f"  • 最优权重: {[f'{w:.1%}' for w in cvxpy_result['weights']]}")

        if cvxopt_result:
            improvement = (cvxopt_result['sharpe'] - current_sharpe) / current_sharpe
            print(f"\n🔧 CVXOPT优化结果:")
            print(f"  • 夏普比率: {cvxopt_result['sharpe']:.2f} (改进{improvement:+.1%})")
            print(f"  • 求解时间: {cvxopt_result['time']:.3f}秒")
            print(f"  • 最优权重: {[f'{w:.1%}' for w in cvxopt_result['weights']]}")

        print("\n💡 实际应用建议:")
        print("  1. 在实际应用中，需要考虑交易成本和市场流动性")
        print("  2. 定期重新平衡投资组合（通常每季度）")
        print("  3. CVXPY适合快速原型开发和学术研究")
        print("  4. CVXOPT适合对性能要求较高的应用")
        print("  5. 对于大规模问题，建议使用商业求解器如MOSEK或Gurobi")

        print("\n⚠️  注意事项:")
        print("  • 历史数据不代表未来表现")
        print("  • 优化结果基于统计假设，存在模型风险")
        print("  • 实际投资需要考虑更多约束和监管要求")


if __name__ == "__main__":
    demo = SimplePortfolioDemo()
    cvxpy_result, cvxopt_result = demo.run_demo()

    print("\n✅ 演示完成！图表已保存到 images/ 目录")