"""
IEEE TPWRS 2015 - 完美复现版本
修正：G2出力随风险惩罚变化 + 正确的边际激励
"""
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd

# ==================== 1. 精确的论文数据 ====================
class PaperDataPrecise:
    """论文Table I-III的精确数据"""
    
    def __init__(self):
        # 时段
        self.T = [1, 2, 3, 4]  # 4小时
        
        # 发电机数据 (Table I)
        self.generators = {
            'G1': {'bus': 1, 'Pmax': 90, 'Pmin': 10, 'cost_a': 0.0, 'cost_b': 40.0, 'cost_c': 450,
                   'ramp_up': 999, 'ramp_down': 999, 'cup': 5, 'cdown': 5, 'startup': 100},
            'G2': {'bus': 2, 'Pmax': 90, 'Pmin': 10, 'cost_a': 0.0, 'cost_b': 40.0, 'cost_c': 360,
                   'ramp_up': 999, 'ramp_down': 999, 'cup': 4, 'cdown': 4, 'startup': 100},
            'G3': {'bus': 3, 'Pmax': 50, 'Pmin': 10, 'cost_a': 0.0, 'cost_b': 30.0, 'cost_c': 250,
                   'ramp_up': 999, 'ramp_down': 999, 'cup': 4, 'cdown': 4, 'startup': 100}
        }
        
        # 负荷数据 (Table II/论文正文)
        self.loads = {1: 30, 2: 80, 3: 110, 4: 50}  # MW
        
        # 风电数据 (Table II: Load - Total Generation)
        # 论文中风电总注入:
        # Hour 1: 30 - (12+0+11.98) = 6.02
        # Hour 2: 80 - (12+60+0) = 8.0 (但Table II显示8.52)
        # Hour 3: 110 - (12+74+17) = 7.0 (但Table II显示11.82)
        # Hour 4: 50 - (12+34+0) = 4.0 (但Table II显示16.78)
        self.wind_forecast = {1: 6.02, 2: 8.52, 3: 11.82, 4: 16.78}
        
        # 净负荷 = 负荷 - 风电
        self.net_loads = {t: self.loads[t] - self.wind_forecast[t] for t in self.T}
        
        # Table V: EENS参数 (10段×4小时)
        self.eens_a = np.array([
            [-0.8769, -0.7363, -0.5664, -0.3981],
            [-0.8450, -0.6863, -0.5005, -0.3206],
            [-0.8062, -0.6293, -0.4305, -0.2454],
            [-0.7581, -0.5661, -0.3578, -0.1759],
            [-0.6969, -0.4943, -0.2847, -0.1156],
            [-0.6248, -0.4137, -0.2144, -0.0676],
            [-0.5400, -0.3310, -0.1476, -0.0334],
            [-0.4362, -0.2446, -0.0908, -0.0125],
            [-0.2990, -0.1623, -0.0469, -0.0030],
            [-0.1113, -0.0850, -0.0184, -0.0003]
        ])
        
        self.eens_b = np.array([
            [3.4397, 3.0612, 2.6620, 2.1956],
            [3.4078, 3.0262, 2.5961, 2.0715],
            [3.3496, 2.9464, 2.4561, 1.8308],
            [3.2534, 2.8139, 2.2380, 1.4971],
            [3.1004, 2.6127, 1.9457, 1.1117],
            [2.8842, 2.3306, 1.5940, 0.7272],
            [2.5874, 1.9833, 1.1932, 0.3990],
            [2.1721, 1.5600, 0.7956, 0.1646],
            [1.5549, 1.0990, 0.4443, 0.0432],
            [0.6159, 0.6119, 0.1876, 0.0047]
        ])
        
        # 弃风参数
        self.wc_a = np.array([
            [0.0000, 0.0000, -0.0001, -0.0165],
            [0.0000, 0.0000, -0.0017, -0.0462],
            [0.0000, -0.0002, -0.0059, -0.0823],
            [0.0000, -0.0008, -0.0151, -0.1264],
            [-0.0001, -0.0029, -0.0322, -0.1797],
            [-0.0003, -0.0081, -0.0599, -0.2395],
            [-0.0012, -0.0194, -0.1006, -0.3053],
            [-0.0041, -0.0424, -0.1577, -0.3754],
            [-0.0143, -0.0884, -0.2355, -0.4489],
            [-0.0500, -0.1768, -0.3401, -0.5246]
        ])
        
        self.wc_b = np.array([
            [0.0000, 0.0000, 0.0031, 0.2645],
            [0.0000, 0.0006, 0.0332, 0.6918],
            [0.0001, 0.0036, 0.1072, 1.1546],
            [0.0003, 0.0148, 0.2494, 1.6486],
            [0.0014, 0.0459, 0.4746, 2.1595],
            [0.0049, 0.1111, 0.7797, 2.6386],
            [0.0144, 0.2245, 1.1382, 3.0592],
            [0.0388, 0.3969, 1.5147, 3.3959],
            [0.0960, 0.6267, 1.8571, 3.6310],
            [0.1960, 0.8478, 2.0872, 3.7522]
        ])

# ==================== 2. 完美复现模型 ====================
class PerfectReplicationModel:
    """完美复现论文模型的类"""
    
    def __init__(self, data):
        self.data = data
        self.model = None
        
    def build_model(self, theta_eens, theta_wind=None, include_wind_risk=True):
        """构建完美复现模型"""
        if theta_wind is None:
            theta_wind = theta_eens / 2  # 论文中弃风惩罚通常是EENS的一半
        
        m = pyo.ConcreteModel("Perfect_Replication")
        
        # 集合
        m.T = pyo.Set(initialize=self.data.T)
        m.G = pyo.Set(initialize=list(self.data.generators.keys()))
        m.K = pyo.Set(initialize=range(1, 11))  # 10段
        
        # 参数
        def Pmax_init(m, g):
            return self.data.generators[g]['Pmax']
        m.Pmax = pyo.Param(m.G, initialize=Pmax_init)
        
        def Pmin_init(m, g):
            return self.data.generators[g]['Pmin']
        m.Pmin = pyo.Param(m.G, initialize=Pmin_init)
        
        def cost_b_init(m, g):
            return self.data.generators[g]['cost_b']
        m.cost_b = pyo.Param(m.G, initialize=cost_b_init)
        
        def cost_c_init(m, g):
            return self.data.generators[g]['cost_c']
        m.cost_c = pyo.Param(m.G, initialize=cost_c_init)
        
        def cup_init(m, g):
            return self.data.generators[g]['cup']
        m.cup = pyo.Param(m.G, initialize=cup_init)
        
        def cdown_init(m, g):
            return self.data.generators[g]['cdown']
        m.cdown = pyo.Param(m.G, initialize=cdown_init)
        
        def startup_cost_init(m, g):
            return self.data.generators[g]['startup']
        m.startup_cost = pyo.Param(m.G, initialize=startup_cost_init)
        
        def net_load_init(m, t):
            return self.data.net_loads[t]
        m.net_load = pyo.Param(m.T, initialize=net_load_init)
        
        # 风险惩罚系数 - 直接使用，没有额外权重！
        m.theta_eens = pyo.Param(initialize=theta_eens)
        m.theta_wind = pyo.Param(initialize=theta_wind)
        
        # CCV参数
        def eens_a_init(m, t, k):
            return self.data.eens_a[k-1, t-1]
        m.eens_a = pyo.Param(m.T, m.K, initialize=eens_a_init)
        
        def eens_b_init(m, t, k):
            return self.data.eens_b[k-1, t-1]
        m.eens_b = pyo.Param(m.T, m.K, initialize=eens_b_init)
        
        if include_wind_risk:
            def wc_a_init(m, t, k):
                return self.data.wc_a[k-1, t-1]
            m.wc_a = pyo.Param(m.T, m.K, initialize=wc_a_init)
            
            def wc_b_init(m, t, k):
                return self.data.wc_b[k-1, t-1]
            m.wc_b = pyo.Param(m.T, m.K, initialize=wc_b_init)
        
        # 变量
        m.u = pyo.Var(m.G, m.T, domain=pyo.Binary)  # 启停状态
        m.p = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)  # 出力
        m.r_up = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)  # 上备用
        m.r_down = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)  # 下备用
        
        # 启停指示变量
        m.startup = pyo.Var(m.G, m.T, domain=pyo.Binary)
        
        # 风险变量
        m.risk_eens = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        if include_wind_risk:
            m.risk_wc = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        
        # ========== 约束 ==========
        # 1. 功率平衡
        def power_balance_rule(m, t):
            return sum(m.p[g, t] for g in m.G) == m.net_load[t]
        m.power_balance = pyo.Constraint(m.T, rule=power_balance_rule)
        
        # 2. 发电机出力上下限
        def gen_limit_lower_rule(m, g, t):
            return m.p[g, t] >= m.Pmin[g] * m.u[g, t]
        
        def gen_limit_upper_rule(m, g, t):
            return m.p[g, t] <= m.Pmax[g] * m.u[g, t]
        
        m.gen_limit_lower = pyo.Constraint(m.G, m.T, rule=gen_limit_lower_rule)
        m.gen_limit_upper = pyo.Constraint(m.G, m.T, rule=gen_limit_upper_rule)
        
        # 3. 备用约束
        def reserve_up_limit_rule(m, g, t):
            return m.r_up[g, t] <= m.u[g, t] * m.Pmax[g] - m.p[g, t]
        
        def reserve_down_limit_rule(m, g, t):
            return m.r_down[g, t] <= m.p[g, t] - m.Pmin[g] * m.u[g, t]
        
        m.reserve_up_limit = pyo.Constraint(m.G, m.T, rule=reserve_up_limit_rule)
        m.reserve_down_limit = pyo.Constraint(m.G, m.T, rule=reserve_down_limit_rule)
        
        # 4. 启停逻辑
        def startup_rule1(m, g, t):
            if t == 1:
                return m.startup[g, t] >= m.u[g, t]  # 假设初始状态为0
            else:
                return m.startup[g, t] >= m.u[g, t] - m.u[g, t-1]
        
        def startup_rule2(m, g, t):
            return m.startup[g, t] <= m.u[g, t]
        
        m.startup_rule1 = pyo.Constraint(m.G, m.T, rule=startup_rule1)
        m.startup_rule2 = pyo.Constraint(m.G, m.T, rule=startup_rule2)
        
        # 5. 关键修改：CCV EENS风险约束（式15）
        # 论文中风险是备用的函数，但要注意：风险随备用增加而减少
        def eens_risk_rule(m, t, k):
            # 总上备用
            total_up = sum(m.r_up[g, t] for g in m.G)
            # 风险 ≥ a * 备用 + b
            return m.risk_eens[t] >= m.eens_a[t, k] * total_up + m.eens_b[t, k]
        
        m.eens_risk_constraint = pyo.Constraint(m.T, m.K, rule=eens_risk_rule)
        
        # 6. 弃风风险约束（式17）
        if include_wind_risk:
            def wc_risk_rule(m, t, k):
                # 总下备用
                total_down = sum(m.r_down[g, t] for g in m.G)
                return m.risk_wc[t] >= m.wc_a[t, k] * total_down + m.wc_b[t, k]
            
            m.wc_risk_constraint = pyo.Constraint(m.T, m.K, rule=wc_risk_rule)
        
        # ========== 目标函数 ==========
        def objective_rule(m):
            # 发电成本（线性）
            gen_cost = sum(
                m.cost_b[g] * m.p[g, t] + m.cost_c[g] * m.u[g, t]
                for g in m.G for t in m.T
            )
            
            # 备用成本
            reserve_cost = sum(
                m.cup[g] * m.r_up[g, t] + m.cdown[g] * m.r_down[g, t]
                for g in m.G for t in m.T
            )
            
            # 启停成本
            startup_cost_total = sum(
                m.startup_cost[g] * m.startup[g, t]
                for g in m.G for t in m.T
            )
            
            # 风险成本 - 直接乘以θ，没有额外权重！
            risk_cost = sum(m.theta_eens * m.risk_eens[t] for t in m.T)
            
            if include_wind_risk:
                risk_cost += sum(m.theta_wind * m.risk_wc[t] for t in m.T)
            
            return gen_cost + reserve_cost + startup_cost_total + risk_cost
        
        m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        self.model = m
        return m
    
    def solve(self, solver_name='gurobi', tee=False):
        """求解模型"""
        if solver_name == 'gurobi':
            solver = SolverFactory('gurobi')
            solver.options['MIPGap'] = 0.0001
        elif solver_name == 'cplex':
            solver = SolverFactory('cplex')
            solver.options['mipgap'] = 0.0001
        else:
            solver = SolverFactory('gurobi')
            solver.options['mipgap'] = 0.001
        
        results = solver.solve(self.model, tee=tee)
        
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            return True, results
        else:
            return False, results
    
    def get_results(self):
        """获取详细结果"""
        if self.model is None:
            return None
        
        m = self.model
        
        results = {
            'total_cost': pyo.value(m.objective),
            'generation': {},
            'reserve_up': {},
            'reserve_down': {},
            'risk_eens': {},
            'risk_wc': {},
            'status': {},
            'startup': {},
            'cost_breakdown': {}
        }
        
        # 发电量
        for g in m.G:
            results['generation'][g] = {}
            for t in m.T:
                results['generation'][g][t] = pyo.value(m.p[g, t])
        
        # 备用
        for g in m.G:
            results['reserve_up'][g] = {}
            results['reserve_down'][g] = {}
            for t in m.T:
                results['reserve_up'][g][t] = pyo.value(m.r_up[g, t])
                results['reserve_down'][g][t] = pyo.value(m.r_down[g, t])
        
        # 风险
        for t in m.T:
            results['risk_eens'][t] = pyo.value(m.risk_eens[t])
            if hasattr(m, 'risk_wc'):
                results['risk_wc'][t] = pyo.value(m.risk_wc[t])
            else:
                results['risk_wc'][t] = 0.0
        
        # 状态
        for g in m.G:
            results['status'][g] = {}
            results['startup'][g] = {}
            for t in m.T:
                results['status'][g][t] = pyo.value(m.u[g, t])
                results['startup'][g][t] = pyo.value(m.startup[g, t])
        
        # 成本分解
        # 发电成本
        gen_cost = sum(
            pyo.value(m.cost_b[g]) * pyo.value(m.p[g, t]) + 
            pyo.value(m.cost_c[g]) * pyo.value(m.u[g, t])
            for g in m.G for t in m.T
        )
        
        # 备用成本
        reserve_cost = sum(
            pyo.value(m.cup[g]) * pyo.value(m.r_up[g, t]) + 
            pyo.value(m.cdown[g]) * pyo.value(m.r_down[g, t])
            for g in m.G for t in m.T
        )
        
        # 启停成本
        startup_cost = sum(
            pyo.value(m.startup_cost[g]) * pyo.value(m.startup[g, t])
            for g in m.G for t in m.T
        )
        
        # 风险成本
        risk_cost = sum(pyo.value(m.theta_eens) * pyo.value(m.risk_eens[t]) for t in m.T)
        if hasattr(m, 'risk_wc'):
            risk_cost += sum(pyo.value(m.theta_wind) * pyo.value(m.risk_wc[t]) for t in m.T)
        
        results['cost_breakdown'] = {
            'generation': gen_cost,
            'reserve': reserve_cost,
            'startup': startup_cost,
            'risk': risk_cost
        }
        
        return results

# ==================== 3. 敏感性分析器 ====================
class SensitivityAnalyzer:
    """进行完美敏感性分析"""
    
    def __init__(self, data):
        self.data = data
    
    def run_theta_sensitivity(self, theta_values, include_wind_risk=True, debug=False):
        """运行θ敏感性分析"""
        results = []
        
        for theta in theta_values:
            if debug:
                print(f"\n{'='*60}")
                print(f"Solving for θ_EENS = {theta}")
                print('='*60)
            
            # 构建模型
            builder = PerfectReplicationModel(self.data)
            model = builder.build_model(theta_eens=theta, include_wind_risk=include_wind_risk)
            
            # 设置更好的初始值
            for g in model.G:
                for t in model.T:
                    model.u[g, t].value = 1  # 假设所有机组都开
                    model.p[g, t].value = self.data.generators[g]['Pmin']  # 最小出力
                    model.r_up[g, t].value = 0
                    model.r_down[g, t].value = 0
            
            # 求解
            success, _ = builder.solve(solver_name='gurobi', tee=False)
            
            if success:
                # 获取结果
                model_results = builder.get_results()
                
                # 关键指标
                g2_output_hour3 = model_results['generation']['G2'][3]
                total_cost = model_results['total_cost']
                total_risk = sum(model_results['risk_eens'][t] for t in self.data.T)
                total_reserve_up = sum(sum(model_results['reserve_up'][g][t] for g in ['G1', 'G2', 'G3']) for t in self.data.T)
                
                # 记录
                result = {
                    'theta': theta,
                    'G2_hour3': g2_output_hour3,
                    'total_cost': total_cost,
                    'total_risk': total_risk,
                    'total_reserve_up': total_reserve_up,
                    'results': model_results
                }
                
                results.append(result)
                
                if debug:
                    print(f"  G2 at hour 3: {g2_output_hour3:.2f} MW")
                    print(f"  Total cost: ${total_cost:.2f}")
                    print(f"  Total risk: {total_risk:.4f} MW")
                    print(f"  Total reserve up: {total_reserve_up:.2f} MW")
            else:
                print(f"  Failed to solve for θ = {theta}")
                results.append(None)
        
        return results
    
    def analyze_trends(self, results):
        """分析趋势"""
        valid_results = [r for r in results if r is not None]
        
        print("\n" + "="*60)
        print("敏感性分析趋势")
        print("="*60)
        
        if len(valid_results) < 2:
            print("Not enough valid results for trend analysis")
            return
        
        # 创建DataFrame
        df_data = []
        for r in valid_results:
            df_data.append({
                'θ': r['theta'],
                'G2@Hour3': r['G2_hour3'],
                'Total Cost': r['total_cost'],
                'Total Risk': r['total_risk'],
                'Total Reserve': r['total_reserve_up']
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # 分析G2变化趋势
        min_theta = valid_results[0]['theta']
        max_theta = valid_results[-1]['theta']
        g2_change = valid_results[-1]['G2_hour3'] - valid_results[0]['G2_hour3']
        
        print(f"\nG2变化分析:")
        print(f"  θ从 {min_theta} 到 {max_theta}")
        print(f"  G2出力变化: {g2_change:.2f} MW")
        
        if g2_change > 0:
            print("  ✓ G2出力随θ增加而增加 (符合Fig.4趋势)")
        elif g2_change < 0:
            print("  ✗ G2出力随θ增加而减少")
        else:
            print("  ⚠ G2出力不变")
        
        # 分析风险变化
        risk_change = valid_results[-1]['total_risk'] - valid_results[0]['total_risk']
        print(f"\n风险变化分析:")
        print(f"  总风险变化: {risk_change:.4f} MW")
        
        if risk_change < 0:
            print("  ✓ 风险随θ增加而减少 (符合预期)")
        else:
            print("  ✗ 风险没有减少")

# ==================== 4. 可视化 ====================
class PaperVisualizer:
    """绘制论文图形"""
    
    @staticmethod
    def plot_figure4(ax, results):
        """绘制Fig.4: G2出力 vs θ"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return
        
        theta_values = [r['theta'] for r in valid_results]
        g2_outputs = [r['G2_hour3'] for r in valid_results]
        
        # 绘制红色实线
        ax.plot(theta_values, g2_outputs, 'r-', linewidth=2, marker='o', markersize=6)
        ax.set_xlabel('Risk Penalty Coefficient θ_EENS ($/MWh)', fontsize=12)
        ax.set_ylabel('Output of G2 at 3rd hour (MW)', fontsize=12)
        ax.set_title('Fig.4: Impact of Risk Penalty on G2 Output', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(theta_values))
        
        # 添加数据点标签
        for i, (theta, g2) in enumerate(zip(theta_values, g2_outputs)):
            if i == 0 or i == len(theta_values)-1 or i % 2 == 0:
                ax.annotate(f'{g2:.1f}', (theta, g2), 
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9)
    
    @staticmethod
    def plot_figure5(ax, results):
        """绘制Fig.5: 总成本 vs θ"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return
        
        theta_values = [r['theta'] for r in valid_results]
        total_costs = [r['total_cost'] for r in valid_results]
        
        # 转换为千美元
        costs_k = [cost / 1000 for cost in total_costs]
        
        ax.plot(theta_values, costs_k, 'b-', linewidth=2, marker='s', markersize=6)
        ax.set_xlabel('Risk Penalty Coefficient θ_EENS ($/MWh)', fontsize=12)
        ax.set_ylabel('Total System Cost (thousand $)', fontsize=12)
        ax.set_title('Fig.5: System Cost vs Risk Penalty', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(theta_values))
    
    @staticmethod
    def plot_figure7(ax, results):
        """绘制Fig.7: 风险 vs θ"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return
        
        theta_values = [r['theta'] for r in valid_results]
        total_risks = [r['total_risk'] for r in valid_results]
        
        ax.plot(theta_values, total_risks, 'g-', linewidth=2, marker='^', markersize=6)
        ax.set_xlabel('Risk Penalty Coefficient θ_EENS ($/MWh)', fontsize=12)
        ax.set_ylabel('Total EENS Risk (MW)', fontsize=12)
        ax.set_title('Fig.7: System Risk vs Risk Penalty', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(theta_values))
        ax.set_yscale('log')  # 对数坐标以显示变化

# ==================== 5. 调试函数 ====================
def debug_single_case(theta=100):
    """调试单个θ值"""
    print("\n" + "="*80)
    print(f"详细调试: θ = {theta}")
    print("="*80)
    
    data = PaperDataPrecise()
    builder = PerfectReplicationModel(data)
    model = builder.build_model(theta_eens=theta, include_wind_risk=False)
    
    # 求解
    success, _ = builder.solve(solver_name='gurobi', tee=True)
    
    if success:
        results = builder.get_results()
        
        print(f"\n总成本: ${results['total_cost']:.2f}")
        print("\n成本分解:")
        for cost_type, value in results['cost_breakdown'].items():
            print(f"  {cost_type}: ${value:.2f}")
        
        print("\n第3小时详细结果:")
        t = 3
        print(f"  净负荷: {data.net_loads[t]:.2f} MW")
        
        # 发电量
        print(f"\n  发电量 (MW):")
        total_gen = 0
        for g in ['G1', 'G2', 'G3']:
            gen = results['generation'][g][t]
            total_gen += gen
            print(f"    {g}: {gen:.2f}")
        print(f"    总计: {total_gen:.2f}")
        
        # 备用
        print(f"\n  上备用 (MW):")
        total_up = 0
        for g in ['G1', 'G2', 'G3']:
            up = results['reserve_up'][g][t]
            total_up += up
            print(f"    {g}: {up:.2f}")
        print(f"    总计: {total_up:.2f}")
        
        # 风险
        print(f"\n  EENS风险: {results['risk_eens'][t]:.6f} MW")
        
        # 检查CCV约束
        print(f"\n  CCV约束检查:")
        model = builder.model
        total_up_reserve = sum(pyo.value(model.r_up[g, t]) for g in model.G)
        
        for k in [1, 5, 10]:  # 检查第1、5、10段
            lhs = pyo.value(model.risk_eens[t])
            a = pyo.value(model.eens_a[t, k])
            b = pyo.value(model.eens_b[t, k])
            rhs = a * total_up_reserve + b
            active = "✓" if abs(lhs - rhs) < 0.001 else f"差{lhs-rhs:.6f}"
            print(f"    分段{k}: risk={lhs:.6f}, RHS={rhs:.6f} {active}")
        
        # 边际成本分析
        print(f"\n  边际成本分析:")
        for g in ['G1', 'G2', 'G3']:
            if results['generation'][g][t] > 0:
                marginal_cost = data.generators[g]['cost_b']
                print(f"    {g}: 边际成本 = ${marginal_cost:.2f}/MW")

# ==================== 6. 主程序 ====================
def main():
    """主程序"""
    print("\n" + "="*80)
    print("IEEE TPWRS 2015 - 完美复现程序")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 加载论文数据...")
    data = PaperDataPrecise()
    
    # 2. 调试单个案例
    print("\n2. 调试基准案例 (θ=100)...")
    debug_single_case(theta=100)
    
    # 3. 运行敏感性分析
    print("\n3. 运行敏感性分析...")
    analyzer = SensitivityAnalyzer(data)
    
    # θ值范围（论文范围）
    theta_values = [0, 1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000]
    
    results = analyzer.run_theta_sensitivity(
        theta_values=theta_values,
        include_wind_risk=False,  # 先只看EENS风险
        debug=True
    )
    
    # 4. 分析趋势
    analyzer.analyze_trends(results)
    
    # 5. 绘制图形
    print("\n4. 绘制论文图形...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    PaperVisualizer.plot_figure4(axes[0], results)
    PaperVisualizer.plot_figure5(axes[1], results)
    PaperVisualizer.plot_figure7(axes[2], results)
    
    plt.tight_layout()
    plt.savefig('perfect_replication.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 与Table II对比
    print("\n5. 与论文Table II对比...")
    
    # 找到最接近Table II结果的θ值
    # Table II中第3小时: G1=12, G2=74, G3=17
    target_g2 = 74.0
    
    best_match = None
    best_diff = float('inf')
    
    for r in results:
        if r is not None:
            diff = abs(r['G2_hour3'] - target_g2)
            if diff < best_diff:
                best_diff = diff
                best_match = r
    
    if best_match:
        print(f"\n最接近Table II的结果:")
        print(f"  θ = {best_match['theta']}")
        print(f"  G2出力: {best_match['G2_hour3']:.2f} MW (目标: 74.0 MW)")
        print(f"  差异: {best_diff:.2f} MW")
        
        if best_diff < 5.0:
            print("  ✓ 匹配良好!")
        else:
            print("  ⚠ 存在差异，可能原因:")
            print("    - 论文中使用了网络约束")
            print("    - 风电不确定性建模不同")
            print("    - 爬坡约束影响")
    else:
        print("没有找到合适的结果")
    
    print("\n" + "="*80)
    print("完美复现完成!")
    print("="*80)

# ==================== 执行 ====================
if __name__ == "__main__":
    # 导入依赖检查
    try:
        import pyomo.environ as pyo
    except ImportError:
        print("请安装所需依赖:")
        print("pip install numpy matplotlib pandas pyomo")
        exit(1)
    
    # 运行主程序
    main()