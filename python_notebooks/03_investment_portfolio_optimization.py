"""
投资组合优化实战案例：使用三大优化包解决实际问题

这个案例模拟了一个基金管理公司的投资组合优化问题，包含：
1. 基础的马科维茨投资组合优化
2. 考虑交易成本的投资组合再平衡
3. 风险预算和约束条件
4. 三个优化包的性能对比
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PortfolioOptimizationCase:
    """
    投资组合优化案例类
    模拟一个实际的基金管理场景
    """
    def __init__(self):
        # 模拟数据：10个资产（股票+债券）
        self.n_assets = 10
        self.asset_names = [
            '科技股', '消费股', '金融股', '医疗股', '工业股',
            '国债', '企业债', '高收益债', 'TIPS', '国际债券'
        ]

        # 生成真实的市场数据特征
        self.annual_returns = np.array([
            0.12, 0.10, 0.08, 0.09, 0.07,  # 股票
            0.04, 0.05, 0.06, 0.03, 0.04   # 债券
        ])

        self.annual_volatilities = np.array([
            0.25, 0.20, 0.18, 0.22, 0.15,  # 股票
            0.08, 0.10, 0.12, 0.06, 0.07   # 债券
        ])

        # 生成相关系数矩阵（模拟真实市场相关性）
        correlations = np.array([
            [1.00, 0.60, 0.40, 0.30, 0.20, 0.10, 0.15, 0.20, 0.05, 0.08],
            [0.60, 1.00, 0.50, 0.35, 0.25, 0.12, 0.18, 0.22, 0.06, 0.10],
            [0.40, 0.50, 1.00, 0.45, 0.30, 0.15, 0.20, 0.25, 0.08, 0.12],
            [0.30, 0.35, 0.45, 1.00, 0.35, 0.18, 0.22, 0.28, 0.10, 0.15],
            [0.20, 0.25, 0.30, 0.35, 1.00, 0.20, 0.25, 0.30, 0.12, 0.18],
            [0.10, 0.12, 0.15, 0.18, 0.20, 1.00, 0.70, 0.40, 0.60, 0.50],
            [0.15, 0.18, 0.20, 0.22, 0.25, 0.70, 1.00, 0.50, 0.55, 0.60],
            [0.20, 0.22, 0.25, 0.28, 0.30, 0.40, 0.50, 1.00, 0.30, 0.35],
            [0.05, 0.06, 0.08, 0.10, 0.12, 0.60, 0.55, 0.30, 1.00, 0.65],
            [0.08, 0.10, 0.12, 0.15, 0.18, 0.50, 0.60, 0.35, 0.65, 1.00]
        ])

        # 计算协方差矩阵
        vol_matrix = np.diag(self.annual_volatilities)
        self.covariance_matrix = vol_matrix @ correlations @ vol_matrix

        # 当前投资组合（模拟现有持仓）
        self.current_weights = np.array([
            0.15, 0.12, 0.10, 0.08, 0.05,  # 股票
            0.20, 0.15, 0.08, 0.04, 0.03   # 债券
        ])

        # 投资约束
        self.min_weight = 0.02  # 最小权重2%
        self.max_weight = 0.30  # 最大权重30%
        self.max_turnover = 0.20  # 最大换手率20%
        self.target_return = 0.08  # 目标收益率8%

        # 交易成本
        self.transaction_cost_rate = 0.002  # 0.2%

        print("=== 投资组合优化案例初始化 ===")
        print(f"资产数量: {self.n_assets}")
        print(f"当前年化收益率: {np.dot(self.current_weights, self.annual_returns):.2%}")
        print(f"当前年化波动率: {np.sqrt(np.dot(self.current_weights, np.dot(self.covariance_matrix, self.current_weights))):.2%}")
        print("-" * 50)

    def solve_with_cvxopt(self):
        """使用CVXOPT求解投资组合优化问题"""
        print("\n=== 使用CVXOPT求解 ===")

        try:
            from cvxopt import matrix, solvers
            start_time = time.time()

            # 构建优化问题
            n = self.n_assets

            # 目标函数：最小化风险
            P = matrix(self.covariance_matrix)
            q = matrix(np.zeros(n))

            # 约束条件
            # 1. 权重约束：min_weight <= w <= max_weight
            G_weight = matrix(np.vstack([-np.eye(n), np.eye(n)]))
            h_weight = matrix(np.concatenate([-self.min_weight * np.ones(n),
                                           self.max_weight * np.ones(n)]))

            # 2. 预算约束：sum(w) = 1
            A_budget = matrix(np.ones(n), (1, n))
            b_budget = matrix(1.0)

            # 3. 收益约束：mu^T * w >= target_return
            A_return = matrix(self.annual_returns, (1, n))
            b_return = matrix(self.target_return)

            # 4. 换手率约束：sum(|w - w_current|) <= max_turnover
            # 通过线性化处理：拆分为买入和卖出
            turnover_vars = 2 * n  # 额外的变量
            n_total = n + turnover_vars

            # 构建完整的QP问题
            P_total = matrix(np.zeros((n_total, n_total)))
            P_total[:n, :n] = self.covariance_matrix

            q_total = matrix(np.concatenate([np.zeros(n),
                                           [self.transaction_cost_rate] * turnover_vars]))

            # 约束矩阵
            G_total = matrix(np.zeros((2*n + 4*n, n_total)))
            h_total = matrix(np.zeros(2*n + 4*n))

            # 权重约束
            G_total[:2*n, :n] = G_weight
            h_total[:2*n] = h_weight

            # 换手率约束
            for i in range(n):
                # 买入变量约束
                G_total[2*n + 4*i, i] = -1
                G_total[2*n + 4*i, n + 2*i] = -1
                G_total[2*n + 4*i + 1, i] = 1
                G_total[2*n + 4*i + 1, n + 2*i] = -1

                # 卖出变量约束
                G_total[2*n + 4*i + 2, i] = 1
                G_total[2*n + 4*i + 2, n + 2*i + 1] = -1
                G_total[2*n + 4*i + 3, i] = -1
                G_total[2*n + 4*i + 3, n + 2*i + 1] = -1

            A_total = matrix(np.vstack([
                np.concatenate([np.ones(n), np.zeros(turnover_vars)]),
                np.concatenate([self.annual_returns, np.zeros(turnover_vars)])
            ]))

            b_total = matrix(np.array([1.0, self.target_return]))

            # 求解
            solvers.options['show_progress'] = False
            solution = solvers.qp(P_total, q_total, G_total, h_total, A_total, b_total)

            solve_time = time.time() - start_time

            if solution['status'] == 'optimal':
                x = np.array(solution['x']).flatten()
                optimal_weights = x[:n]

                # 计算性能指标
                portfolio_return = np.dot(optimal_weights, self.annual_returns)
                portfolio_risk = np.sqrt(np.dot(optimal_weights,
                                               np.dot(self.covariance_matrix, optimal_weights)))
                turnover = np.sum(np.abs(optimal_weights - self.current_weights))

                print(f"求解成功！用时: {solve_time:.3f}秒")
                print(f"预期收益率: {portfolio_return:.2%}")
                print(f"预期波动率: {portfolio_risk:.2%}")
                print(f"夏普比率: {portfolio_return/portfolio_risk:.2f}")
                print(f"换手率: {turnover:.2%}")

                return {
                    'weights': optimal_weights,
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe': portfolio_return/portfolio_risk,
                    'turnover': turnover,
                    'solve_time': solve_time,
                    'status': 'optimal'
                }
            else:
                print(f"求解失败: {solution['status']}")
                return {'status': 'failed', 'reason': solution['status']}

        except ImportError:
            print("CVXOPT未安装，跳过")
            return {'status': 'not_installed'}
        except Exception as e:
            print(f"CVXOPT求解出错: {e}")
            return {'status': 'error', 'reason': str(e)}

    def solve_with_mosek(self):
        """使用MOSEK求解投资组合优化问题"""
        print("\n=== 使用MOSEK求解 ===")

        try:
            from mosek.fusion import Model, ObjectiveSense, Expr, Domain
            start_time = time.time()

            with Model('Portfolio Optimization') as M:
                # 变量定义
                w = M.variable('w', self.n_assets, Domain.inRange(self.min_weight, self.max_weight))

                # 目标函数：最小化风险
                portfolio_risk = Expr.dot(w, Expr.mul(self.covariance_matrix, w))
                M.objective(ObjectiveSense.Minimize, portfolio_risk)

                # 约束条件
                M.constraint('budget', Expr.sum(w), Domain.equalsTo(1.0))
                M.constraint('return', Expr.dot(self.annual_returns, w),
                           Domain.greaterThan(self.target_return))

                # 换手率约束（简化为L1范数约束）
                turnover = Expr.sum(Expr.abs(Expr.sub(w, self.current_weights)))
                M.constraint('turnover', turnover, Domain.lessThan(self.max_turnover))

                # 求解
                M.solve()

                solve_time = time.time() - start_time

                if M.getPrimalSolutionStatus() in [ProblemStatus.PrimalAndDualFeasible]:
                    optimal_weights = w.level()

                    # 计算性能指标
                    portfolio_return = np.dot(optimal_weights, self.annual_returns)
                    portfolio_risk = np.sqrt(np.dot(optimal_weights,
                                                   np.dot(self.covariance_matrix, optimal_weights)))
                    turnover = np.sum(np.abs(optimal_weights - self.current_weights))

                    print(f"求解成功！用时: {solve_time:.3f}秒")
                    print(f"预期收益率: {portfolio_return:.2%}")
                    print(f"预期波动率: {portfolio_risk:.2%}")
                    print(f"夏普比率: {portfolio_return/portfolio_risk:.2f}")
                    print(f"换手率: {turnover:.2%}")

                    return {
                        'weights': optimal_weights,
                        'return': portfolio_return,
                        'risk': portfolio_risk,
                        'sharpe': portfolio_return/portfolio_risk,
                        'turnover': turnover,
                        'solve_time': solve_time,
                        'status': 'optimal'
                    }
                else:
                    print(f"求解失败: {M.getPrimalSolutionStatus()}")
                    return {'status': 'failed', 'reason': str(M.getPrimalSolutionStatus())}

        except ImportError:
            print("MOSEK未安装，跳过")
            return {'status': 'not_installed'}
        except Exception as e:
            print(f"MOSEK求解出错: {e}")
            return {'status': 'error', 'reason': str(e)}

    def solve_with_gurobi(self):
        """使用Gurobi求解投资组合优化问题"""
        print("\n=== 使用Gurobi求解 ===")

        try:
            import gurobipy as gp
            from gurobipy import GRB
            start_time = time.time()

            # 创建模型
            model = gp.Model('Portfolio Optimization')

            # 添加变量
            w = model.addVars(self.n_assets, lb=self.min_weight, ub=self.max_weight,
                            name='weights')

            # 目标函数：最小化风险
            risk_expr = gp.QuadExpr()
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    risk_expr += self.covariance_matrix[i,j] * w[i] * w[j]

            model.setObjective(risk_expr, GRB.MINIMIZE)

            # 约束条件
            model.addConstr(gp.quicksum(w[i] for i in range(self.n_assets)) == 1, 'budget')
            model.addConstr(gp.quicksum(self.annual_returns[i] * w[i] for i in range(self.n_assets)) >=
                           self.target_return, 'return')

            # 换手率约束
            turnover_expr = gp.quicksum(abs(w[i] - self.current_weights[i]) for i in range(self.n_assets))
            model.addConstr(turnover_expr <= self.max_turnover, 'turnover')

            # 设置参数
            model.Params.TimeLimit = 60  # 60秒时间限制
            model.Params.OutputFlag = 0  # 关闭输出

            # 求解
            model.optimize()

            solve_time = time.time() - start_time

            if model.status == GRB.OPTIMAL:
                optimal_weights = [w[i].X for i in range(self.n_assets)]

                # 计算性能指标
                portfolio_return = np.dot(optimal_weights, self.annual_returns)
                portfolio_risk = np.sqrt(np.dot(optimal_weights,
                                               np.dot(self.covariance_matrix, optimal_weights)))
                turnover = np.sum(np.abs(optimal_weights - self.current_weights))

                print(f"求解成功！用时: {solve_time:.3f}秒")
                print(f"预期收益率: {portfolio_return:.2%}")
                print(f"预期波动率: {portfolio_risk:.2%}")
                print(f"夏普比率: {portfolio_return/portfolio_risk:.2f}")
                print(f"换手率: {turnover:.2%}")

                return {
                    'weights': optimal_weights,
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe': portfolio_return/portfolio_risk,
                    'turnover': turnover,
                    'solve_time': solve_time,
                    'status': 'optimal'
                }
            else:
                print(f"求解失败: {model.status}")
                return {'status': 'failed', 'reason': str(model.status)}

        except ImportError:
            print("Gurobi未安装，跳过")
            return {'status': 'not_installed'}
        except Exception as e:
            print(f"Gurobi求解出错: {e}")
            return {'status': 'error', 'reason': str(e)}

    def compare_results(self, results):
        """比较三个优化包的结果"""
        print("\n" + "="*60)
        print("=== 优化包性能对比 ===")
        print("="*60)

        valid_results = {k: v for k, v in results.items() if v.get('status') == 'optimal'}

        if not valid_results:
            print("没有成功求解的结果")
            return

        # 创建对比表格
        comparison_data = []
        for solver_name, result in valid_results.items():
            comparison_data.append([
                solver_name,
                f"{result['return']:.2%}",
                f"{result['risk']:.2%}",
                f"{result['sharpe']:.2f}",
                f"{result['turnover']:.2%}",
                f"{result['solve_time']:.3f}s"
            ])

        headers = ['优化包', '预期收益率', '预期波动率', '夏普比率', '换手率', '求解时间']
        print(f"{'优化包':<12} {'预期收益率':<12} {'预期波动率':<12} {'夏普比率':<10} {'换手率':<10} {'求解时间':<10}")
        print("-" * 70)

        for data in comparison_data:
            print(f"{data[0]:<12} {data[1]:<12} {data[2]:<12} {data[3]:<10} {data[4]:<10} {data[5]:<10}")

        # 可视化对比
        self.visualize_comparison(valid_results)

    def visualize_comparison(self, results):
        """可视化对比结果"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        solvers = list(results.keys())
        returns = [results[s]['return'] for s in solvers]
        risks = [results[s]['risk'] for s in solvers]
        sharpe_ratios = [results[s]['sharpe'] for s in solvers]
        solve_times = [results[s]['solve_time'] for s in solvers]

        # 1. 风险-收益散点图
        ax1.scatter(risks, returns, s=100, alpha=0.7)
        for i, solver in enumerate(solvers):
            ax1.annotate(solver, (risks[i], returns[i]), xytext=(5, 5),
                        textcoords='offset points', fontsize=10)
        ax1.set_xlabel('风险 (波动率)')
        ax1.set_ylabel('收益率')
        ax1.set_title('风险-收益对比')
        ax1.grid(True, alpha=0.3)

        # 2. 夏普比率对比
        bars1 = ax2.bar(solvers, sharpe_ratios, alpha=0.7)
        ax2.set_ylabel('夏普比率')
        ax2.set_title('夏普比率对比')
        ax2.grid(True, alpha=0.3, axis='y')
        for bar, ratio in zip(bars1, sharpe_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.2f}', ha='center', va='bottom')

        # 3. 求解时间对比
        bars2 = ax3.bar(solvers, solve_times, alpha=0.7)
        ax3.set_ylabel('求解时间 (秒)')
        ax3.set_title('求解性能对比')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, time_val in zip(bars2, solve_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom')

        # 4. 投资组合权重对比
        if len(results) > 0:
            first_solver = list(results.keys())[0]
            weights = results[first_solver]['weights']

            ax4.pie(weights, labels=self.asset_names, autopct='%1.1f%%', startangle=90)
            ax4.set_title(f'{first_solver} 最优投资组合权重')

        plt.tight_layout()
        plt.savefig('../images/portfolio_optimization_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_comprehensive_analysis(self):
        """运行完整的投资组合优化分析"""
        print("开始投资组合优化综合分析...")
        print("="*60)

        # 展示当前投资组合
        current_return = np.dot(self.current_weights, self.annual_returns)
        current_risk = np.sqrt(np.dot(self.current_weights, np.dot(self.covariance_matrix, self.current_weights)))
        current_sharpe = current_return / current_risk

        print(f"当前投资组合表现:")
        print(f"预期收益率: {current_return:.2%}")
        print(f"预期波动率: {current_risk:.2%}")
        print(f"夏普比率: {current_sharpe:.2f}")
        print("-" * 60)

        # 使用三个优化包求解
        results = {}

        # CVXOPT
        cvxopt_result = self.solve_with_cvxopt()
        if cvxopt_result['status'] == 'optimal':
            results['CVXOPT'] = cvxopt_result

        # MOSEK
        mosek_result = self.solve_with_mosek()
        if mosek_result['status'] == 'optimal':
            results['MOSEK'] = mosek_result

        # Gurobi
        gurobi_result = self.solve_with_gurobi()
        if gurobi_result['status'] == 'optimal':
            results['Gurobi'] = gurobi_result

        # 对比结果
        self.compare_results(results)

        # 生成详细报告
        self.generate_report(results)

        return results

    def generate_report(self, results):
        """生成优化报告"""
        print("\n" + "="*60)
        print("=== 投资组合优化报告 ===")
        print("="*60)

        valid_results = {k: v for k, v in results.items() if v.get('status') == 'optimal'}

        if valid_results:
            best_sharpe = max(valid_results.items(), key=lambda x: x[1]['sharpe'])
            fastest = min(valid_results.items(), key=lambda x: x[1]['solve_time'])

            print(f"最佳夏普比率: {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.2f})")
            print(f"最快求解: {fastest[0]} ({fastest[1]['solve_time']:.3f}秒)")

            # 改进幅度
            current_return = np.dot(self.current_weights, self.annual_returns)
            current_risk = np.sqrt(np.dot(self.current_weights, np.dot(self.covariance_matrix, self.current_weights)))

            for solver_name, result in valid_results.items():
                improvement = (result['sharpe'] - current_return/current_risk) / (current_return/current_risk)
                print(f"{solver_name} 夏普比率改进: {improvement:+.1%}")

        print("\n投资建议:")
        print("1. 根据优化结果，建议调整投资组合权重")
        print("2. 考虑交易成本和换手率限制")
        print("3. 定期重新平衡投资组合")
        print("4. 监控市场变化并及时调整策略")


if __name__ == "__main__":
    # 创建并运行投资组合优化案例
    portfolio_case = PortfolioOptimizationCase()
    results = portfolio_case.run_comprehensive_analysis()

    print("\n分析完成！生成的可视化图表已保存到 ../images/ 目录")