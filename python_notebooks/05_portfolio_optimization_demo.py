"""
投资组合优化实战演示

使用模拟数据展示三个优化包在投资组合优化中的实际应用
这个演示完整地展示了从数据处理到优化求解的全过程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PortfolioOptimizationDemo:
    """
    投资组合优化演示类
    展示CVXPY、CVXOPT等优化包的实际应用
    """
    def __init__(self):
        # 模拟真实的投资场景
        self.n_assets = 8
        self.asset_names = [
            '科技股ETF', '消费股ETF', '医疗股ETF', '金融股ETF',
            '国债ETF', '公司债ETF', '黄金ETF', '房地产ETF'
        ]

        # 生成符合真实市场特征的数据
        self.generate_market_data()

        # 投资约束条件
        self.min_weight = 0.05  # 最小5%
        self.max_weight = 0.35  # 最大35%
        self.max_turnover = 0.30  # 最大换手率30%
        self.target_return = 0.08  # 目标收益率8%

        # 当前投资组合（模拟实际持仓）
        self.current_weights = np.array([
            0.20, 0.15, 0.12, 0.08,  # 股票ETF
            0.25, 0.12, 0.05, 0.03   # 债券和其他
        ])

        print("=== 投资组合优化实战演示 ===")
        print(f"资产数量: {self.n_assets}")
        print(f"分析期间: 3年历史数据")
        print("-" * 60)

    def generate_market_data(self):
        """生成符合真实市场特征的数据"""
        np.random.seed(42)

        # 设置真实的年化收益率和波动率
        annual_returns = np.array([
            0.12, 0.10, 0.09, 0.08,   # 股票ETF
            0.04, 0.05, 0.06, 0.07    # 债券和其他
        ])

        annual_vols = np.array([
            0.22, 0.18, 0.16, 0.15,   # 股票ETF
            0.06, 0.08, 0.15, 0.12    # 债券和其他
        ])

        # 生成相关系数矩阵（模拟真实市场相关性）
        correlations = np.array([
            [1.00, 0.70, 0.50, 0.40, 0.10, 0.15, 0.20, 0.25],
            [0.70, 1.00, 0.60, 0.45, 0.12, 0.18, 0.22, 0.28],
            [0.50, 0.60, 1.00, 0.55, 0.15, 0.20, 0.25, 0.30],
            [0.40, 0.45, 0.55, 1.00, 0.18, 0.22, 0.28, 0.32],
            [0.10, 0.12, 0.15, 0.18, 1.00, 0.70, 0.30, 0.35],
            [0.15, 0.18, 0.20, 0.22, 0.70, 1.00, 0.35, 0.40],
            [0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 1.00, 0.50],
            [0.25, 0.28, 0.30, 0.32, 0.35, 0.40, 0.50, 1.00]
        ])

        # 计算协方差矩阵
        vol_matrix = np.diag(annual_vols)
        self.covariance_matrix = vol_matrix @ correlations @ vol_matrix
        self.annual_returns = annual_returns

        # 生成历史价格数据（3年日度数据）
        n_days = 756  # 3年
        daily_returns = np.zeros((n_days, self.n_assets))

        # 生成相关的日收益率
        for day in range(n_days):
            # 生成多元正态分布的随机数
            cholesky_decomp = np.linalg.cholesky(self.covariance_matrix / 252)
            random_shocks = np.random.randn(self.n_assets)
            daily_returns[day] = annual_returns / 252 + cholesky_decomp @ random_shocks

        self.daily_returns = pd.DataFrame(daily_returns)

        # 生成价格序列
        self.prices = pd.DataFrame(100 * np.cumprod(1 + daily_returns, axis=0),
                                  columns=self.asset_names)

        # 添加日期索引
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        self.daily_returns.index = dates
        self.prices.index = dates

        print("生成的资产特征:")
        for i, name in enumerate(self.asset_names):
            print(f"  {name}: 年化收益{annual_returns[i]:.1%}, 波动率{annual_vols[i]:.1%}")

    def solve_with_cvxpy(self):
        """使用CVXPY求解投资组合优化"""
        print("\n=== 使用CVXPY求解 ===")

        try:
            import cvxpy as cp
            start_time = time.time()

            # 定义优化变量
            w = cp.Variable(self.n_assets)

            # 目标函数：最小化风险（方差）
            portfolio_risk = cp.quad_form(w, self.covariance_matrix)
            portfolio_return = self.annual_returns @ w

            # 约束条件
            constraints = [
                cp.sum(w) == 1,                              # 投资权重和为1
                w >= self.min_weight,                       # 最小权重限制
                w <= self.max_weight,                       # 最大权重限制
                portfolio_return >= self.target_return,      # 最低收益要求
                cp.norm(w - self.current_weights, 1) <= self.max_turnover  # 换手率限制
            ]

            # 构建并求解优化问题
            problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
            # 尝试不同的求解器
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except:
                try:
                    problem.solve(solver=cp.SCS, verbose=False)
                except:
                    problem.solve(verbose=False)

            solve_time = time.time() - start_time

            if problem.status == 'optimal':
                optimal_weights = w.value

                # 计算性能指标
                portfolio_return_val = np.dot(optimal_weights, self.annual_returns)
                portfolio_risk_val = np.sqrt(np.dot(optimal_weights,
                                                   np.dot(self.covariance_matrix, optimal_weights)))
                turnover = np.sum(np.abs(optimal_weights - self.current_weights))
                sharpe_ratio = portfolio_return_val / portfolio_risk_val

                print(f"✓ 求解成功！用时: {solve_time:.3f}秒")
                print(f"  预期年化收益率: {portfolio_return_val:.2%}")
                print(f"  预期年化波动率: {portfolio_risk_val:.2%}")
                print(f"  夏普比率: {sharpe_ratio:.2f}")
                print(f"  换手率: {turnover:.2%}")

                return {
                    'weights': optimal_weights,
                    'return': portfolio_return_val,
                    'risk': portfolio_risk_val,
                    'sharpe': sharpe_ratio,
                    'turnover': turnover,
                    'solve_time': solve_time,
                    'status': 'optimal'
                }
            else:
                print(f"✗ 求解失败: {problem.status}")
                return {'status': 'failed', 'reason': problem.status}

        except ImportError:
            print("✗ CVXPY未安装，跳过")
            return {'status': 'not_installed'}
        except Exception as e:
            print(f"✗ CVXPY求解出错: {e}")
            return {'status': 'error', 'reason': str(e)}

    def solve_with_cvxopt(self):
        """使用CVXOPT求解投资组合优化"""
        print("\n=== 使用CVXOPT求解 ===")

        try:
            from cvxopt import matrix, solvers
            start_time = time.time()

            # 转换为CVXOPT矩阵格式
            n = self.n_assets

            # 目标函数：min 0.5 * w^T * Σ * w
            P = matrix(self.covariance_matrix)
            q = matrix(np.zeros(n))

            # 不等式约束：G * w <= h
            # 包含：w >= min_weight, w <= max_weight
            G_ineq = matrix(np.vstack([-np.eye(n), np.eye(n)]))
            h_ineq = matrix(np.concatenate([-self.min_weight * np.ones(n),
                                           self.max_weight * np.ones(n)]))

            # 等式约束：A * w = b
            # 包含：sum(w) = 1, mu^T * w >= target_return
            A_eq = matrix(np.vstack([np.ones(n), self.annual_returns]))
            b_eq = matrix(np.array([1.0, self.target_return]))

            # 求解
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G_ineq, h_ineq, A_eq, b_eq)

            solve_time = time.time() - start_time

            if solution['status'] == 'optimal':
                optimal_weights = np.array(solution['x']).flatten()

                # 计算性能指标
                portfolio_return = np.dot(optimal_weights, self.annual_returns)
                portfolio_risk = np.sqrt(np.dot(optimal_weights,
                                               np.dot(self.covariance_matrix, optimal_weights)))
                turnover = np.sum(np.abs(optimal_weights - self.current_weights))
                sharpe_ratio = portfolio_return / portfolio_risk

                print(f"✓ 求解成功！用时: {solve_time:.3f}秒")
                print(f"  预期年化收益率: {portfolio_return:.2%}")
                print(f"  预期年化波动率: {portfolio_risk:.2%}")
                print(f"  夏普比率: {sharpe_ratio:.2f}")
                print(f"  换手率: {turnover:.2%}")

                return {
                    'weights': optimal_weights,
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'sharpe': sharpe_ratio,
                    'turnover': turnover,
                    'solve_time': solve_time,
                    'status': 'optimal'
                }
            else:
                print(f"✗ 求解失败: {solution['status']}")
                return {'status': 'failed', 'reason': solution['status']}

        except ImportError:
            print("✗ CVXOPT未安装，跳过")
            return {'status': 'not_installed'}
        except Exception as e:
            print(f"✗ CVXOPT求解出错: {e}")
            return {'status': 'error', 'reason': str(e)}

    def calculate_efficient_frontier(self):
        """计算有效前沿"""
        print("\n=== 计算有效前沿 ===")

        try:
            import cvxpy as cp

            target_returns = np.linspace(0.04, 0.15, 25)
            frontier_risks = []
            frontier_returns = []
            frontier_weights = []

            for target_return in target_returns:
                w = cp.Variable(self.n_assets)
                portfolio_risk = cp.quad_form(w, self.covariance_matrix)
                portfolio_return = self.annual_returns @ w

                constraints = [
                    cp.sum(w) == 1,
                    w >= self.min_weight,
                    w <= self.max_weight,
                    portfolio_return >= target_return
                ]

                problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
                # 尝试不同的求解器
                try:
                    problem.solve(solver=cp.ECOS, verbose=False)
                except:
                    try:
                        problem.solve(solver=cp.SCS, verbose=False)
                    except:
                        problem.solve(verbose=False)

                if problem.status == 'optimal':
                    frontier_risks.append(np.sqrt(problem.value))
                    frontier_returns.append(target_return)
                    frontier_weights.append(w.value)

            return np.array(frontier_risks), np.array(frontier_returns), np.array(frontier_weights)

        except ImportError:
            print("CVXPY未安装，无法计算有效前沿")
            return None, None, None

    def analyze_portfolio_performance(self, weights, name="投资组合"):
        """分析投资组合的历史表现"""
        # 计算投资组合日收益率
        portfolio_daily_returns = self.daily_returns @ weights

        # 计算累计收益
        cumulative_returns = (1 + portfolio_daily_returns).cumprod()

        # 计算性能指标
        annual_return = portfolio_daily_returns.mean() * 252
        annual_vol = portfolio_daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

        # 计算Sortino比率
        downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
        sortino_ratio = annual_return / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0

        print(f"\n{name}历史表现分析:")
        print(f"  年化收益率: {annual_return:.2%}")
        print(f"  年化波动率: {annual_vol:.2%}")
        print(f"  夏普比率: {sharpe_ratio:.2f}")
        print(f"  Sortino比率: {sortino_ratio:.2f}")
        print(f"  最大回撤: {max_drawdown:.2%}")

        return {
            'cumulative_returns': cumulative_returns,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'daily_returns': portfolio_daily_returns
        }

    def create_comprehensive_visualization(self, results, efficient_frontier_data=None):
        """创建综合可视化图表"""
        fig = plt.figure(figsize=(20, 16))

        # 1. 有效前沿和优化结果对比
        ax1 = plt.subplot(3, 3, 1)
        if efficient_frontier_data is not None:
            risks, returns, weights = efficient_frontier_data
            ax1.plot(risks, returns, 'b-', linewidth=2, label='有效前沿')

            # 标记最小方差组合
            min_var_idx = np.argmin(risks)
            ax1.scatter(risks[min_var_idx], returns[min_var_idx], c='red', s=100,
                       marker='o', label='最小方差组合')

        # 标记各资产
        asset_risks = np.sqrt(np.diag(self.covariance_matrix))
        ax1.scatter(asset_risks, self.annual_returns, c='lightgray', s=80,
                   alpha=0.8, label='单个资产')

        # 标记当前组合
        current_risk = np.sqrt(np.dot(self.current_weights,
                                    np.dot(self.covariance_matrix, self.current_weights)))
        current_return = np.dot(self.current_weights, self.annual_returns)
        ax1.scatter(current_risk, current_return, c='red', s=150,
                   marker='s', label='当前组合', edgecolors='darkred', linewidth=2)

        # 标记优化结果
        valid_results = {k: v for k, v in results.items() if v.get('status') == 'optimal'}
        colors = ['blue', 'green', 'purple']
        for i, (solver_name, result) in enumerate(valid_results.items()):
            ax1.scatter(result['risk'], result['return'], c=colors[i], s=150,
                       marker='^', label=f'{solver_name}优化', edgecolors='black', linewidth=1)

        ax1.set_xlabel('风险 (年化波动率)')
        ax1.set_ylabel('收益 (年化收益率)')
        ax1.set_title('投资组合优化结果')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 权重分布对比
        ax2 = plt.subplot(3, 3, 2)
        x = np.arange(self.n_assets)
        width = 0.35

        bars_current = ax2.bar(x - width/2, self.current_weights, width,
                               label='当前组合', alpha=0.8, color='lightcoral')

        if 'CVXPY' in valid_results:
            bars_optimal = ax2.bar(x + width/2, valid_results['CVXPY']['weights'], width,
                                   label='CVXPY优化', alpha=0.8, color='lightblue')

        ax2.set_xlabel('资产')
        ax2.set_ylabel('投资权重')
        ax2.set_title('投资组合权重对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name[:4] for name in self.asset_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加权重值标签
        for i, (curr, opt) in enumerate(zip(self.current_weights, valid_results.get('CVXPY', {}).get('weights', self.current_weights))):
            ax2.text(i - width/2, curr + 0.005, f'{curr:.1%}', ha='center', va='bottom', fontsize=8)
            if 'CVXPY' in valid_results:
                ax2.text(i + width/2, opt + 0.005, f'{opt:.1%}', ha='center', va='bottom', fontsize=8)

        # 3. 资产价格走势
        ax3 = plt.subplot(3, 3, 3)
        normalized_prices = self.prices / self.prices.iloc[0]
        for i, name in enumerate(self.asset_names[:5]):  # 显示前5个资产
            ax3.plot(normalized_prices.index, normalized_prices.values[:, i],
                    label=name, alpha=0.8, linewidth=1.5)

        ax3.set_ylabel('标准化价格')
        ax3.set_title('资产价格走势 (前5个资产)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 相关性热图
        ax4 = plt.subplot(3, 3, 4)
        correlation_matrix = self.daily_returns.corr()
        im = ax4.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')

        # 添加相关性数值
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)

        ax4.set_xticks(range(self.n_assets))
        ax4.set_yticks(range(self.n_assets))
        ax4.set_xticklabels([name[:4] for name in self.asset_names], rotation=45)
        ax4.set_yticklabels([name[:4] for name in self.asset_names])
        ax4.set_title('资产相关性矩阵')
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

        # 5. 滚动波动率
        ax5 = plt.subplot(3, 3, 5)
        window_size = 63  # 3个月
        rolling_vol = self.daily_returns.rolling(window=window_size).std() * np.sqrt(252)

        for i, name in enumerate(self.asset_names[:4]):  # 显示前4个资产
            ax5.plot(rolling_vol.index, rolling_vol.values[:, i], label=name, alpha=0.8)

        ax5.set_ylabel('滚动波动率 (年化)')
        ax5.set_title(f'滚动波动率 ({window_size}天)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 收益率分布
        ax6 = plt.subplot(3, 3, 6)
        current_returns = self.daily_returns @ self.current_weights

        # 绘制当前组合的收益分布
        ax6.hist(current_returns, bins=40, alpha=0.7, label='当前组合',
                density=True, color='lightcoral', edgecolor='black')

        if 'CVXPY' in valid_results:
            optimal_returns = self.daily_returns @ valid_results['CVXPY']['weights']
            ax6.hist(optimal_returns, bins=40, alpha=0.7, label='优化组合',
                    density=True, color='lightblue', edgecolor='black')

        ax6.set_xlabel('日收益率')
        ax6.set_ylabel('概率密度')
        ax6.set_title('收益率分布对比')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. 累计收益曲线
        ax7 = plt.subplot(3, 3, 7)
        if 'CVXPY' in valid_results:
            current_perf = self.analyze_portfolio_performance(self.current_weights, "当前组合")
            optimal_perf = self.analyze_portfolio_performance(valid_results['CVXPY']['weights'], "优化组合")

            ax7.plot(current_perf['cumulative_returns'].index, current_perf['cumulative_returns'],
                    label='当前组合', color='lightcoral', linewidth=2)
            ax7.plot(optimal_perf['cumulative_returns'].index, optimal_perf['cumulative_returns'],
                    label='优化组合', color='lightblue', linewidth=2)

        ax7.set_ylabel('累计收益')
        ax7.set_title('累计收益曲线')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. 回撤分析
        ax8 = plt.subplot(3, 3, 8)
        if 'CVXPY' in valid_results:
            # 计算回撤
            current_cumulative = current_perf['cumulative_returns']
            optimal_cumulative = optimal_perf['cumulative_returns']

            current_drawdown = (current_cumulative / current_cumulative.cummax() - 1)
            optimal_drawdown = (optimal_cumulative / optimal_cumulative.cummax() - 1)

            ax8.fill_between(current_drawdown.index, current_drawdown, 0,
                           alpha=0.5, label='当前组合回撤', color='lightcoral')
            ax8.fill_between(optimal_drawdown.index, optimal_drawdown, 0,
                           alpha=0.5, label='优化组合回撤', color='lightblue')

        ax8.set_ylabel('回撤率')
        ax8.set_title('回撤分析')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. 风险收益散点图（不同资产配置）
        ax9 = plt.subplot(3, 3, 9)
        # 生成随机权重组合
        np.random.seed(42)
        n_random = 1000
        random_weights = np.random.dirichlet(np.ones(self.n_assets), n_random)

        random_returns = random_weights @ self.annual_returns
        random_risks = np.sqrt([np.dot(w, np.dot(self.covariance_matrix, w))
                               for w in random_weights])

        ax9.scatter(random_risks, random_returns, c='lightgray', s=10, alpha=0.6, label='随机组合')

        # 标记当前和优化组合
        ax9.scatter(current_risk, current_return, c='red', s=200, marker='s',
                   label='当前组合', edgecolors='darkred', linewidth=2)

        if 'CVXPY' in valid_results:
            result = valid_results['CVXPY']
            ax9.scatter(result['risk'], result['return'], c='blue', s=200, marker='^',
                       label='优化组合', edgecolors='darkblue', linewidth=2)

        ax9.set_xlabel('风险 (年化波动率)')
        ax9.set_ylabel('收益 (年化收益率)')
        ax9.set_title('可行投资组合分布')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../images/portfolio_optimization_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_demo(self):
        """运行完整的投资组合优化演示"""
        print("开始投资组合优化实战演示...")
        print("="*60)

        # 分析当前投资组合表现
        current_performance = self.analyze_portfolio_performance(self.current_weights, "当前投资组合")

        # 使用不同优化包求解
        results = {}

        # CVXPY
        cvxpy_result = self.solve_with_cvxpy()
        if cvxpy_result['status'] == 'optimal':
            results['CVXPY'] = cvxpy_result

        # CVXOPT
        cvxopt_result = self.solve_with_cvxopt()
        if cvxopt_result['status'] == 'optimal':
            results['CVXOPT'] = cvxopt_result

        # 计算有效前沿
        efficient_frontier_data = self.calculate_efficient_frontier()

        # 创建综合可视化
        self.create_comprehensive_visualization(results, efficient_frontier_data)

        # 生成详细报告
        self.generate_final_report(results, current_performance)

        return results

    def generate_final_report(self, results, current_performance):
        """生成最终优化报告"""
        print("\n" + "="*60)
        print("=== 投资组合优化最终报告 ===")
        print("="*60)

        print("📊 投资组合概况:")
        print(f"  • 资产数量: {self.n_assets}")
        print(f"  • 最小权重: {self.min_weight:.1%}")
        print(f"  • 最大权重: {self.max_weight:.1%}")
        print(f"  • 最大换手率: {self.max_turnover:.1%}")
        print(f"  • 目标收益率: {self.target_return:.1%}")

        print(f"\n📈 当前投资组合表现:")
        print(f"  • 年化收益率: {current_performance['annual_return']:.2%}")
        print(f"  • 年化波动率: {current_performance['annual_vol']:.2%}")
        print(f"  • 夏普比率: {current_performance['sharpe_ratio']:.2f}")
        print(f"  • Sortino比率: {current_performance['sortino_ratio']:.2f}")
        print(f"  • 最大回撤: {current_performance['max_drawdown']:.2%}")

        valid_results = {k: v for k, v in results.items() if v.get('status') == 'optimal'}
        if valid_results:
            print(f"\n🎯 优化结果对比:")

            for solver_name, result in valid_results.items():
                print(f"\n  {solver_name}:")
                print(f"    • 预期年化收益率: {result['return']:.2%}")
                print(f"    • 预期年化波动率: {result['risk']:.2%}")
                print(f"    • 夏普比率: {result['sharpe']:.2f}")
                print(f"    • 换手率: {result['turnover']:.1%}")
                print(f"    • 求解时间: {result['solve_time']:.3f}秒")

                # 计算改进幅度
                sharpe_improvement = (result['sharpe'] - current_performance['sharpe_ratio']) / current_performance['sharpe_ratio']
                print(f"    • 夏普比率改进: {sharpe_improvement:+.1%}")

            # 找出最佳方案
            best_solver = max(valid_results.items(), key=lambda x: x[1]['sharpe'])
            print(f"\n🏆 推荐方案: {best_solver[0]}")
            print(f"    • 最高夏普比率: {best_solver[1]['sharpe']:.2f}")

        print(f"\n💡 投资建议:")
        print("  1. 根据优化结果重新平衡投资组合")
        print("  2. 注意控制交易成本和税收影响")
        print("  3. 建议每季度重新评估和调整")
        print("  4. 密切关注市场变化，特别是黑天鹅事件")
        print("  5. 根据个人风险承受能力调整目标收益率")

        print(f"\n⚠️  风险提示:")
        print("  • 历史表现不代表未来收益")
        print("  • 优化结果基于历史数据和统计假设")
        print("  • 市场存在不确定性，需做好风险管理")
        print("  • 建议咨询专业投资顾问")

        print("\n" + "="*60)


if __name__ == "__main__":
    # 运行投资组合优化演示
    demo = PortfolioOptimizationDemo()
    results = demo.run_demo()

    print("\n✅ 投资组合优化演示完成！")
    print("📁 可视化图表已保存到 ../images/ 目录")
    print("🔍 详细分析结果请查看控制台输出")