"""
真实数据投资组合优化案例

使用真实的金融市场数据（股票、ETF等）进行投资组合优化，
展示三个优化包在真实场景中的应用和性能对比。
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class RealPortfolioOptimization:
    """
    真实数据投资组合优化类
    使用Yahoo Finance获取真实市场数据
    """
    def __init__(self):
        # 定义投资组合（包含不同类型的资产）
        self.tickers = [
            'AAPL',  # 苹果 - 科技股
            'MSFT',  # 微软 - 科技股
            'JPM',   # 摩根大通 - 金融股
            'JNJ',   # 强生 - 医疗股
            'XOM',   # 埃克森美孚 - 能源股
            'SPY',   # SPDR标普500ETF - 大盘股
            'TLT',   # iShares 20+年期国债ETF - 长期债券
            'LQD',   # iShares投资级公司债ETF - 公司债
            'GLD',   # SPDR黄金ETF - 黄金
            'VNQ'    # 先锋房地产ETF - 房地产
        ]

        self.asset_names = [
            '苹果', '微软', '摩根大通', '强生', '埃克森美孚',
            '标普500ETF', '长期国债ETF', '公司债ETF', '黄金ETF', '房地产ETF'
        ]

        # 获取历史数据（过去3年的日度数据）
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=3*365)

        print("=== 真实数据投资组合优化案例 ===")
        print("正在获取历史市场数据...")

        self.fetch_data()
        self.calculate_parameters()

        # 投资约束
        self.min_weight = 0.02  # 最小权重2%
        self.max_weight = 0.40  # 最大权重40%
        self.max_turnover = 0.25  # 最大换手率25%
        self.target_return = 0.10  # 目标年化收益率10%

        # 当前投资组合（模拟现有持仓）
        self.current_weights = np.array([
            0.15, 0.15, 0.10, 0.10, 0.05,  # 股票
            0.20, 0.15, 0.05, 0.03, 0.02   # 其他资产
        ])

        print(f"数据获取完成，包含 {len(self.returns)} 个交易日")
        print(f"资产数量: {self.n_assets}")
        print("-" * 60)

    def fetch_data(self):
        """获取Yahoo Finance数据"""
        try:
            # 获取调整后的收盘价
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']

            # 处理缺失值
            data = data.fillna(method='ffill')  # 前向填充
            data = data.dropna(axis=1)  # 删除仍有缺失值的列

            # 计算日收益率
            self.returns = data.pct_change().dropna()

            # 更新有效资产的tickers和names
            self.tickers = list(data.columns)
            self.asset_names = [self.asset_names[self.tickers.index(t)] if t in self.tickers else t for t in data.columns]
            self.n_assets = len(self.tickers)

            self.prices = data

            print("成功获取以下资产数据:")
            for i, (ticker, name) in enumerate(zip(self.tickers, self.asset_names)):
                print(f"  {i+1}. {ticker} ({name})")

        except Exception as e:
            print(f"数据获取失败: {e}")
            print("使用模拟数据...")
            # 使用模拟数据作为备选
            self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """生成模拟市场数据作为备选"""
        print("使用模拟数据...")

        np.random.seed(42)
        n_days = 756  # 3年约252个交易日
        self.n_assets = len(self.tickers)

        # 生成随机价格序列
        annual_returns = np.random.normal(0.08, 0.15, self.n_assets)
        annual_vols = np.random.uniform(0.15, 0.35, self.n_assets)

        returns_matrix = np.zeros((n_days, self.n_assets))
        for i in range(self.n_assets):
            daily_returns = np.random.normal(annual_returns[i]/252, annual_vols[i]/np.sqrt(252), n_days)
            returns_matrix[:, i] = daily_returns

        self.returns = pd.DataFrame(returns_matrix, columns=self.tickers)

        # 生成价格数据
        self.prices = 100 * (1 + self.returns).cumprod()

    def calculate_parameters(self):
        """计算投资组合参数"""
        # 年化收益率
        self.annual_returns = self.returns.mean() * 252

        # 年化协方差矩阵
        self.covariance_matrix = self.returns.cov() * 252

        # 确保协方差矩阵正定
        eigenvalues = np.linalg.eigvals(self.covariance_matrix)
        if np.min(eigenvalues) < 1e-8:
            # 添加小的正则化项
            self.covariance_matrix += np.eye(self.n_assets) * 1e-6

        print(f"年化收益率范围: {self.annual_returns.min():.2%} ~ {self.annual_returns.max():.2%}")
        print(f"年化波动率范围: {np.sqrt(np.diag(self.covariance_matrix)).min():.2%} ~ {np.sqrt(np.diag(self.covariance_matrix)).max():.2%}")

    def solve_with_cvxpy(self):
        """使用CVXPY求解（如果安装了优化后端）"""
        print("\n=== 使用CVXPY求解 ===")

        try:
            import cvxpy as cp
            start_time = time.time()

            # 定义变量
            w = cp.Variable(self.n_assets)

            # 目标函数：最小化风险
            portfolio_risk = cp.quad_form(w, self.covariance_matrix)
            portfolio_return = self.annual_returns.values @ w

            # 约束条件
            constraints = [
                cp.sum(w) == 1,  # 预算约束
                w >= self.min_weight,  # 最小权重
                w <= self.max_weight,  # 最大权重
                portfolio_return >= self.target_return,  # 收益约束
                cp.norm(w - self.current_weights, 1) <= self.max_turnover  # 换手率约束
            ]

            # 构建问题
            problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)

            # 求解
            problem.solve(solver=cp.ECOS, verbose=False)

            solve_time = time.time() - start_time

            if problem.status == 'optimal':
                optimal_weights = w.value

                # 计算性能指标
                portfolio_return_val = np.dot(optimal_weights, self.annual_returns)
                portfolio_risk_val = np.sqrt(np.dot(optimal_weights,
                                                   np.dot(self.covariance_matrix, optimal_weights)))
                turnover = np.sum(np.abs(optimal_weights - self.current_weights))

                print(f"求解成功！用时: {solve_time:.3f}秒")
                print(f"预期年化收益率: {portfolio_return_val:.2%}")
                print(f"预期年化波动率: {portfolio_risk_val:.2%}")
                print(f"夏普比率: {portfolio_return_val/portfolio_risk_val:.2f}")
                print(f"换手率: {turnover:.2%}")

                return {
                    'weights': optimal_weights,
                    'return': portfolio_return_val,
                    'risk': portfolio_risk_val,
                    'sharpe': portfolio_return_val/portfolio_risk_val,
                    'turnover': turnover,
                    'solve_time': solve_time,
                    'status': 'optimal'
                }
            else:
                print(f"求解失败: {problem.status}")
                return {'status': 'failed', 'reason': problem.status}

        except ImportError:
            print("CVXPY未安装，跳过")
            return {'status': 'not_installed'}
        except Exception as e:
            print(f"CVXPY求解出错: {e}")
            return {'status': 'error', 'reason': str(e)}

    def solve_with_cvxopt(self):
        """使用CVXOPT求解"""
        print("\n=== 使用CVXOPT求解 ===")

        try:
            from cvxopt import matrix, solvers
            start_time = time.time()

            # 构建优化问题
            n = self.n_assets

            # 目标函数：最小化风险
            P = matrix(self.covariance_matrix.values)
            q = matrix(np.zeros(n))

            # 约束条件
            # 1. 权重约束
            G_weight = matrix(np.vstack([-np.eye(n), np.eye(n)]))
            h_weight = matrix(np.concatenate([-self.min_weight * np.ones(n),
                                           self.max_weight * np.ones(n)]))

            # 2. 预算约束和收益约束
            A = matrix(np.vstack([np.ones(n), self.annual_returns.values]))
            b = matrix(np.array([1.0, self.target_return]))

            # 3. 换手率约束（简化处理）
            # 使用L1范数约束的线性化
            G_turnover = np.zeros((2*n, n))
            h_turnover = np.zeros(2*n)
            for i in range(n):
                G_turnover[2*i, i] = 1
                G_turnover[2*i+1, i] = -1
                h_turnover[2*i] = self.current_weights[i] + self.max_turnover/2
                h_turnover[2*i+1] = -self.current_weights[i] + self.max_turnover/2

            G_total = matrix(np.vstack([G_weight, G_turnover]))
            h_total = matrix(np.concatenate([h_weight, h_turnover]))

            # 求解
            solvers.options['show_progress'] = False
            solution = solvers.qp(P, q, G_total, h_total, A, b)

            solve_time = time.time() - start_time

            if solution['status'] == 'optimal':
                optimal_weights = np.array(solution['x']).flatten()

                # 计算性能指标
                portfolio_return = np.dot(optimal_weights, self.annual_returns)
                portfolio_risk = np.sqrt(np.dot(optimal_weights,
                                               np.dot(self.covariance_matrix, optimal_weights)))
                turnover = np.sum(np.abs(optimal_weights - self.current_weights))

                print(f"求解成功！用时: {solve_time:.3f}秒")
                print(f"预期年化收益率: {portfolio_return:.2%}")
                print(f"预期年化波动率: {portfolio_risk:.2%}")
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

    def calculate_efficient_frontier(self):
        """计算有效前沿"""
        print("\n=== 计算有效前沿 ===")

        try:
            import cvxpy as cp

            target_returns = np.linspace(0.05, 0.20, 20)
            risks = []
            returns = []
            weights_list = []

            for target_return in target_returns:
                w = cp.Variable(self.n_assets)
                portfolio_risk = cp.quad_form(w, self.covariance_matrix.values)
                portfolio_return = self.annual_returns.values @ w

                constraints = [
                    cp.sum(w) == 1,
                    w >= self.min_weight,
                    w <= self.max_weight,
                    portfolio_return >= target_return
                ]

                problem = cp.Problem(cp.Minimize(portfolio_risk), constraints)
                problem.solve(solver=cp.ECOS, verbose=False)

                if problem.status == 'optimal':
                    risks.append(np.sqrt(problem.value))
                    returns.append(target_return)
                    weights_list.append(w.value)

            return np.array(risks), np.array(returns), np.array(weights_list)

        except ImportError:
            print("CVXPY未安装，无法计算有效前沿")
            return None, None, None

    def analyze_historical_performance(self, weights, name="投资组合"):
        """分析历史表现"""
        portfolio_returns = self.returns @ weights

        # 计算累计收益
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # 计算性能指标
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

        print(f"\n{name}历史表现分析:")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"年化波动率: {annual_vol:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")

        return {
            'cumulative_returns': cumulative_returns,
            'annual_return': annual_return,
            'annual_vol': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def visualize_results(self, results, efficient_frontier_data=None):
        """可视化优化结果"""
        fig = plt.figure(figsize=(20, 15))

        # 1. 有效前沿和优化结果
        ax1 = plt.subplot(2, 3, 1)
        if efficient_frontier_data is not None:
            risks, returns, _ = efficient_frontier_data
            ax1.plot(risks, returns, 'b-', linewidth=2, label='有效前沿')

        # 绘制各资产
        asset_risks = np.sqrt(np.diag(self.covariance_matrix))
        ax1.scatter(asset_risks, self.annual_returns, c='lightgray', s=50, alpha=0.7, label='单个资产')

        # 绘制当前投资组合
        current_risk = np.sqrt(np.dot(self.current_weights,
                                    np.dot(self.covariance_matrix, self.current_weights)))
        current_return = np.dot(self.current_weights, self.annual_returns)
        ax1.scatter(current_risk, current_return, c='red', s=100, marker='s', label='当前组合')

        # 绘制优化结果
        valid_results = {k: v for k, v in results.items() if v.get('status') == 'optimal'}
        colors = ['blue', 'green', 'orange']
        for i, (solver_name, result) in enumerate(valid_results.items()):
            ax1.scatter(result['risk'], result['return'], c=colors[i], s=100,
                       marker='^', label=f'{solver_name}优化')

        ax1.set_xlabel('风险 (年化波动率)')
        ax1.set_ylabel('收益 (年化收益率)')
        ax1.set_title('投资组合优化结果')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 权重对比
        ax2 = plt.subplot(2, 3, 2)
        x = np.arange(self.n_assets)
        width = 0.25

        ax2.bar(x - width, self.current_weights, width, label='当前组合', alpha=0.7)
        if 'CVXPY' in valid_results:
            ax2.bar(x, valid_results['CVXPY']['weights'], width, label='CVXPY优化', alpha=0.7)

        ax2.set_xlabel('资产')
        ax2.set_ylabel('权重')
        ax2.set_title('投资组合权重对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels([name[:4] for name in self.asset_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 历史价格走势
        ax3 = plt.subplot(2, 3, 3)
        normalized_prices = self.prices / self.prices.iloc[0]
        for i, ticker in enumerate(self.tickers[:5]):  # 只显示前5个资产
            ax3.plot(normalized_prices.index, normalized_prices[ticker],
                    label=self.asset_names[i], alpha=0.7)
        ax3.set_ylabel('标准化价格')
        ax3.set_title('资产价格走势')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 相关性热图
        ax4 = plt.subplot(2, 3, 4)
        correlation_matrix = self.returns.corr()
        im = ax4.imshow(correlation_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax4.set_xticks(range(self.n_assets))
        ax4.set_yticks(range(self.n_assets))
        ax4.set_xticklabels([name[:4] for name in self.asset_names], rotation=45)
        ax4.set_yticklabels([name[:4] for name in self.asset_names])
        ax4.set_title('资产相关性矩阵')
        plt.colorbar(im, ax=ax4)

        # 5. 滚动波动率
        ax5 = plt.subplot(2, 3, 5)
        rolling_vol = self.returns.rolling(window=63).std() * np.sqrt(252)  # 3个月滚动波动率
        for i, ticker in enumerate(self.tickers[:5]):
            ax5.plot(rolling_vol.index, rolling_vol[ticker], label=self.asset_names[i], alpha=0.7)
        ax5.set_ylabel('滚动波动率 (年化)')
        ax5.set_title('滚动波动率 (3个月)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. 收益分布
        ax6 = plt.subplot(2, 3, 6)
        portfolio_returns_current = self.returns @ self.current_weights
        ax6.hist(portfolio_returns_current, bins=50, alpha=0.7, label='当前组合', density=True)

        if 'CVXPY' in valid_results:
            portfolio_returns_optimal = self.returns @ valid_results['CVXPY']['weights']
            ax6.hist(portfolio_returns_optimal, bins=50, alpha=0.7, label='优化组合', density=True)

        ax6.set_xlabel('日收益率')
        ax6.set_ylabel('密度')
        ax6.set_title('收益率分布')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../images/real_portfolio_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_analysis(self):
        """运行完整的分析"""
        print("开始真实数据投资组合优化分析...")
        print("="*60)

        # 显示当前投资组合表现
        current_performance = self.analyze_historical_performance(self.current_weights, "当前投资组合")

        # 使用不同优化包求解
        results = {}

        # CVXPY
        cvxpy_result = self.solve_with_cvxpy()
        if cvxpy_result['status'] == 'optimal':
            results['CVXPY'] = cvxpy_result
            optimal_performance = self.analyze_historical_performance(cvxpy_result['weights'], "优化投资组合")

        # CVXOPT
        cvxopt_result = self.solve_with_cvxopt()
        if cvxopt_result['status'] == 'optimal':
            results['CVXOPT'] = cvxopt_result

        # 计算有效前沿
        efficient_frontier_data = self.calculate_efficient_frontier()

        # 可视化结果
        self.visualize_results(results, efficient_frontier_data)

        # 生成报告
        self.generate_report(results, current_performance, optimal_performance if 'optimal_performance' in locals() else None)

        return results

    def generate_report(self, results, current_perf, optimal_perf=None):
        """生成详细报告"""
        print("\n" + "="*60)
        print("=== 真实数据投资组合优化报告 ===")
        print("="*60)

        print(f"分析期间: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"资产数量: {self.n_assets}")
        print("-" * 60)

        print("当前投资组合表现:")
        print(f"  年化收益率: {current_perf['annual_return']:.2%}")
        print(f"  年化波动率: {current_perf['annual_vol']:.2%}")
        print(f"  夏普比率: {current_perf['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {current_perf['max_drawdown']:.2%}")

        if optimal_perf:
            print("\n优化投资组合表现:")
            print(f"  年化收益率: {optimal_perf['annual_return']:.2%}")
            print(f"  年化波动率: {optimal_perf['annual_vol']:.2%}")
            print(f"  夏普比率: {optimal_perf['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {optimal_perf['max_drawdown']:.2%}")

            improvement = (optimal_perf['sharpe_ratio'] - current_perf['sharpe_ratio']) / current_perf['sharpe_ratio']
            print(f"\n夏普比率改进: {improvement:+.1%}")

        # 优化包对比
        valid_results = {k: v for k, v in results.items() if v.get('status') == 'optimal'}
        if len(valid_results) > 0:
            print("\n优化包性能对比:")
            for solver_name, result in valid_results.items():
                print(f"  {solver_name}:")
                print(f"    求解时间: {result['solve_time']:.3f}秒")
                print(f"    预期夏普比率: {result['sharpe']:.2f}")

        print("\n投资建议:")
        print("1. 基于优化结果重新平衡投资组合")
        print("2. 考虑交易成本和市场流动性")
        print("3. 定期监控和重新平衡（建议每季度）")
        print("4. 注意控制风险，特别是在市场波动较大时")
        print("5. 分散投资于不同资产类别以降低组合风险")


if __name__ == "__main__":
    # 创建并运行真实数据投资组合优化
    real_portfolio = RealPortfolioOptimization()
    results = real_portfolio.run_analysis()

    print("\n分析完成！生成的可视化图表已保存到 ../images/ 目录")