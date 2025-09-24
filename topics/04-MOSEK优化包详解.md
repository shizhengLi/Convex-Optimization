# MOSEK优化包详解

## 1. MOSEK简介

MOSEK是一个高性能的凸优化求解器，特别擅长解决锥优化问题。在金融行业中，MOSEK被广泛用于投资组合优化、风险管理和衍生品定价等领域。

### 1.1 核心优势
- **锥优化专家**：在二阶锥优化(SOCP)和半定优化(SDP)方面表现卓越
- **数值稳定性**：即使在病态条件下也能提供精确解
- **大规模问题**：能够高效处理数百万个变量和约束的问题
- **灵敏度分析**：提供详细的对偶变量和灵敏度信息

### 1.2 支持的问题类型
- 线性规划 (LP)
- 二次规划 (QP/QCQP)
- 二阶锥规划 (SOCP)
- 半定规划 (SDP)
- 几何规划 (GP)
- 指数锥优化

## 2. MOSEK基础用法

### 2.1 安装和许可证
```python
# 安装MOSEK
pip install mosek

# 获取学术许可证（免费）
import mosek
# 需要申请学术许可证并配置
```

### 2.2 基本语法结构
```python
import mosek
from mosek.fusion import *

# 创建模型
with Model('portfolio') as M:
    # 定义变量
    x = M.variable('x', n, Domain.greaterThan(0.0))

    # 设置目标函数
    M.objective(ObjectiveSense.Minimize, Expr.dot(q, x))

    # 添加约束
    M.constraint(Expr.sum(x), Domain.equalsTo(1.0))
    M.constraint(Expr.dot(mu, x), Domain.greaterThan(target_return))

    # 求解
    M.solve()

    # 获取结果
    solution = x.level()
```

## 3. 金融应用案例

### 3.1 马科维茨投资组合优化

#### 问题描述
给定n个资产的期望收益率和协方差矩阵，寻找最小风险的投资组合。

#### MOSEK实现
```python
import mosek
from mosek.fusion import *
import numpy as np

def markowitz_portfolio_mosek(mu, Sigma, target_return):
    """
    使用MOSEK求解马科维茨投资组合优化

    参数:
    mu: 期望收益率向量
    Sigma: 协方差矩阵
    target_return: 目标收益率

    返回:
    w: 最优权重
    risk: 投资组合风险
    """
    n = len(mu)

    with Model('Markowitz Portfolio') as M:
        # 决策变量：投资权重
        w = M.variable('w', n, Domain.greaterThan(0.0))

        # 目标函数：最小化风险
        M.objective(ObjectiveSense.Minimize,
                   Expr.dot(w, Matrix.dense(Sigma) @ w))

        # 约束条件
        M.constraint('budget', Expr.sum(w), Domain.equalsTo(1.0))
        M.constraint('return', Expr.dot(mu, w),
                    Domain.greaterThan(target_return))

        # 求解
        M.solve()

        # 获取结果
        optimal_weights = w.level()
        portfolio_risk = M.primalObjValue()

        return optimal_weights, portfolio_risk
```

### 3.2 风险平价投资组合

#### 问题描述
风险平价(Risk Parity)要求每个资产对总风险的贡献相等。

#### MOSEK实现
```python
def risk_parity_mosek(Sigma):
    """
    使用MOSEK求解风险平价投资组合

    参数:
    Sigma: 协方差矩阵

    返回:
    w: 最优权重
    """
    n = Sigma.shape[0]

    with Model('Risk Parity') as M:
        # 决策变量
        w = M.variable('w', n, Domain.greaterThan(0.0))
        risk_contribution = M.variable('rc', n, Domain.unbounded())

        # 计算总风险
        portfolio_risk = Expr.dot(w, Matrix.dense(Sigma) @ w)

        # 风险贡献约束：每个资产的风险贡献相等
        for i in range(n):
            # 第i个资产的风险贡献 = w_i * (Sigma @ w)_i
            marginal_risk = Expr.dot(Matrix.dense(Sigma[i,:]), w)
            M.constraint(Expr.mul(w.index(i), marginal_risk) -
                       risk_contribution.index(i), Domain.equalsTo(0.0))

        # 所有风险贡献相等
        for i in range(1, n):
            M.constraint(risk_contribution.index(i) -
                       risk_contribution.index(0), Domain.equalsTo(0.0))

        # 预算约束
        M.constraint(Expr.sum(w), Domain.equalsTo(1.0))

        # 求解
        M.solve()

        return w.level()
```

### 3.3 带交易成本的投资组合优化

#### 问题描述
在投资组合再平衡时考虑交易成本的影响。

#### MOSEK实现
```python
def portfolio_with_transaction_costs_mosek(current_w, mu, Sigma,
                                         target_return, transaction_cost_rate):
    """
    带交易成本的投资组合优化

    参数:
    current_w: 当前投资组合权重
    mu: 期望收益率向量
    Sigma: 协方差矩阵
    target_return: 目标收益率
    transaction_cost_rate: 交易成本率

    返回:
    new_w: 新的投资组合权重
    """
    n = len(mu)

    with Model('Portfolio with Transaction Costs') as M:
        # 决策变量
        new_w = M.variable('new_w', n, Domain.greaterThan(0.0))

        # 交易量（买入为正，卖出为负）
        trade = Expr.sub(new_w, current_w)

        # 交易成本（线性近似）
        buy_cost = M.variable('buy_cost', n, Domain.greaterThan(0.0))
        sell_cost = M.variable('sell_cost', n, Domain.greaterThan(0.0))

        # 计算交易成本
        for i in range(n):
            M.constraint(buy_cost.index(i), Domain.greaterThan(trade.index(i)))
            M.constraint(sell_cost.index(i), Domain.greaterThan(
                        Expr.neg(trade.index(i))))

        total_cost = Expr.mul(transaction_cost_rate,
                             Expr.add(Expr.sum(buy_cost), Expr.sum(sell_cost)))

        # 目标函数：最小化风险减去净收益
        net_return = Expr.dot(mu, new_w) - total_cost
        M.objective(ObjectiveSense.Maximize,
                   Expr.sub(net_return, Expr.dot(new_w, Matrix.dense(Sigma) @ new_w)))

        # 约束条件
        M.constraint(Expr.sum(new_w), Domain.equalsTo(1.0))
        M.constraint(Expr.dot(mu, new_w), Domain.greaterThan(target_return))

        # 求解
        M.solve()

        return new_w.level()
```

## 4. 高级特性

### 4.1 锥优化示例

#### 二阶锥优化（SOCP）
```python
def socp_example_mosek():
    """二阶锥优化示例：最小化投资组合的CVaR"""

    with Model('SOCP Example') as M:
        # 变量
        w = M.variable('w', n, Domain.greaterThan(0.0))
        t = M.variable('t', 1, Domain.unbounded())

        # 二阶锥约束：||Sigma^{1/2} w||_2 <= t
        # 这等价于 w^T Sigma w <= t^2
        sqrt_Sigma = np.linalg.cholesky(Sigma)
        M.constraint(Expr.vstack(t, Matrix.dense(sqrt_Sigma) @ w),
                    Domain.inQCone())

        # 目标函数和约束
        M.objective(ObjectiveSense.Minimize, t)
        M.constraint(Expr.sum(w), Domain.equalsTo(1.0))
        M.constraint(Expr.dot(mu, w), Domain.greaterThan(target_return))

        M.solve()
        return w.level()
```

### 4.2 参数化求解

#### 有效前沿计算
```python
def efficient_frontier_mosek(mu, Sigma, return_targets):
    """
    计算投资组合的有效前沿

    参数:
    mu: 期望收益率向量
    Sigma: 协方差矩阵
    return_targets: 目标收益率列表

    返回:
    risks: 对应的风险值
    weights: 对应的投资组合权重
    """
    risks = []
    weights_list = []

    with Model('Efficient Frontier') as M:
        # 变量
        w = M.variable('w', len(mu), Domain.greaterThan(0.0))

        # 固定约束
        M.constraint(Expr.sum(w), Domain.equalsTo(1.0))

        # 参数化目标收益率约束
        return_con = M.constraint(Expr.dot(mu, w), Domain.equalsTo(0.0))

        # 目标：最小化风险
        M.objective(ObjectiveSense.Minimize,
                   Expr.dot(w, Matrix.dense(Sigma) @ w))

        # 求解不同目标收益率下的最优投资组合
        for target in return_targets:
            # 更新目标收益率约束
            return_con.update(0.0, target)

            # 重新求解
            M.solve()

            risks.append(np.sqrt(M.primalObjValue()))
            weights_list.append(w.level())

    return np.array(risks), np.array(weights_list)
```

### 4.3 不确定性优化

#### 鲁棒投资组合优化
```python
def robust_portfolio_mosek(mu, Sigma, uncertainty_set):
    """
    鲁棒投资组合优化：处理期望收益率的不确定性

    参数:
    mu: 名义期望收益率向量
    Sigma: 协方差矩阵
    uncertainty_set: 不确定性集合参数
    """
    n = len(mu)

    with Model('Robust Portfolio') as M:
        # 变量
        w = M.variable('w', n, Domain.greaterThan(0.0))
        z = M.variable('z', n, Domain.unbounded())  # 不确定性变量

        # 鲁棒约束：对于所有不确定性，收益率都满足要求
        M.constraint('worst_case_return',
                     Expr.dot(mu, w) + uncertainty_set * Expr.sum(z),
                     Domain.greaterThan(target_return))

        # 不确定性约束
        M.constraint(Expr.hstack(z, w), Domain.inQCone())

        # 目标函数
        M.objective(ObjectiveSense.Minimize,
                   Expr.dot(w, Matrix.dense(Sigma) @ w))

        # 预算约束
        M.constraint(Expr.sum(w), Domain.equalsTo(1.0))

        M.solve()
        return w.level()
```

## 5. 性能调优技巧

### 5.1 求解器参数设置
```python
def optimize_mosek_parameters(M):
    """优化MOSEK求解器参数"""

    # 设置求解器参数
    M.setSolverParam('numThreads', 4)  # 使用4个线程
    M.setSolverParam('intpntCoTolPfeas', 1e-8)  # 原始可行性容差
    M.setSolverParam('intpntCoTolDfeas', 1e-8)  # 对偶可行性容差
    M.setSolverParam('intpntTolPgap', 1e-8)     # 间隙容差

    # 对于大规模问题，可以开启内存节省模式
    M.setSolverParam('intpntSolveForm', 'intpnt')  # 使用原始-对偶内点法
```

### 5.2 问题重构技巧
- **避免稠密矩阵**：使用稀疏矩阵表示约束系数
- **变量边界**：设置合理的变量边界
- **热启动**：对于参数化问题，使用前一次解作为初始点

## 6. 实际应用建议

### 6.1 金融行业最佳实践
1. **数据预处理**：确保协方差矩阵是正定的
2. **数值稳定性**：对极端值进行处理
3. **结果验证**：检查解的合理性和经济意义
4. **性能监控**：记录求解时间和迭代次数

### 6.2 常见问题解决
- **数值不收敛**：调整容差参数或问题重构
- **内存不足**：使用稀疏矩阵或分解大规模问题
- **许可证问题**：确保学术许可证正确配置

MOSEK在金融优化中的应用非常广泛，特别是在需要高精度求解复杂凸优化问题的场景中表现出色。通过合理使用其丰富的功能和调优选项，可以高效解决各种实际的金融优化问题。