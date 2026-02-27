"""
IEEE TPWRS 2015 - 完整复现版本（含网络约束）
修复：添加网络约束 + 正确的G2出力计算
"""
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 精确的论文数据（添加网络数据） ====================
class PaperDataPrecise:
    """论文Table I-III的精确数据，包含网络约束"""
    
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
        
        # 网络数据（三母线系统，论文图2）
        # 支路数据：from_bus, to_bus, reactance, capacity (MW)
        # 根据论文，支路容量为55MW（图6说明）
        self.branches = {
            'L1': {'from': 1, 'to': 2, 'reactance': 0.1, 'capacity': 55},
            'L2': {'from': 2, 'to': 3, 'reactance': 0.1, 'capacity': 55},
            'L3': {'from': 1, 'to': 3, 'reactance': 0.1, 'capacity': 55}
        }
        
        # PTDF矩阵（功率传输分布因子）- 简化计算
        # 假设所有支路电抗相同
        self.ptdf = self.calculate_ptdf()
        
        # 负荷数据 (Table II/论文正文)
        self.loads = {1: 30, 2: 80, 3: 110, 4: 50}  # MW
        
        # 风电数据（根据论文描述）
        # 母线上风电场：W1在母线1，W2在母线2
        self.wind_farms = {
            'W1': {'bus': 1, 'forecast': {1: 3.0, 2: 4.26, 3: 5.91, 4: 8.39}},  # 50% of total
            'W2': {'bus': 2, 'forecast': {1: 3.0, 2: 4.26, 3: 5.91, 4: 8.39}}   # 50% of total
        }
        
        # 计算净负荷（负荷 - 风电总预测）
        self.net_loads = {}
        for t in self.T:
            total_wind = sum(self.wind_farms[w]['forecast'][t] for w in self.wind_farms)
            self.net_loads[t] = self.loads[t] - total_wind
        
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
        
        # 支路溢出风险参数（表VI）- 简化版本
        # 我们使用支路3的正向参数
        self.flow_a = np.array([
            [0.0165, 0.1483, 0.3755, 0.0842],
            [0.9117, 8.3147, 21.3359, 4.8007],
            [0.0, 0.0, 0.0, 0.0],  # 填充
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
        
        self.flow_b = np.array([
            [0.0009, 0.0186, 0.2040, 0.0661],
            [0.0505, 1.0428, 11.8004, 0.9052],
            [0.0, 0.0, 0.0, 0.0],  # 填充
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
    
    def calculate_ptdf(self):
        """计算三母线系统的PTDF矩阵"""
        # 简化计算：假设所有支路电抗相同
        # 对于三母线系统，PTDF可以解析计算
        ptdf = {}
        
        # 支路L1（母线1-2）的PTDF
        ptdf['L1'] = {
            'G1': 0.5,   # 从G1注入到支路L1的比例
            'G2': -0.5,  # 从G2注入到支路L1的比例
            'G3': 0.0,   # 从G3注入到支路L1的比例
            'load': 0.0  # 负荷对支路的影响
        }
        
        # 支路L2（母线2-3）的PTDF
        ptdf['L2'] = {
            'G1': 0.25,
            'G2': 0.5,
            'G3': -0.75,
            'load': 0.75  # 负荷在母线3
        }
        
        # 支路L3（母线1-3）的PTDF
        ptdf['L3'] = {
            'G1': 0.75,
            'G2': 0.25,
            'G3': -0.25,
            'load': 0.25
        }
        
        return ptdf
    
    def get_branch_flow(self, generation, load, wind):
        """计算支路潮流"""
        flows = {}
        
        for branch_id, branch_data in self.branches.items():
            ptdf = self.ptdf[branch_id]
            
            # 发电机注入贡献
            gen_flow = 0
            for gen_id, gen_val in generation.items():
                gen_bus = self.generators[gen_id]['bus']
                # 简化：使用PTDF
                gen_flow += ptdf.get(f'G{gen_bus}', 0) * gen_val
            
            # 负荷贡献（负荷为负注入）
            load_flow = -ptdf['load'] * load
            
            # 风电贡献
            wind_flow = 0
            for wind_id, wind_val in wind.items():
                wind_bus = self.wind_farms[wind_id]['bus']
                wind_flow += ptdf.get(f'G{wind_bus}', 0) * wind_val
            
            flows[branch_id] = gen_flow + load_flow + wind_flow
        
        return flows

# ==================== 2. 完整模型（含网络约束） ====================
class CompletePaperModel:
    """完整论文模型，包含网络约束"""
    
    def __init__(self, data):
        self.data = data
        self.model = None
        
    def build_model(self, theta_eens=50, theta_wind=None, theta_flow=None, include_flow_risk=True):
        """构建完整模型"""
        if theta_wind is None:
            theta_wind = theta_eens  # 论文中通常设为相等
        if theta_flow is None:
            theta_flow = theta_eens  # 论文中通常设为相等
        
        m = pyo.ConcreteModel("Complete_Replication")
        
        # 集合
        m.T = pyo.Set(initialize=self.data.T)
        m.G = pyo.Set(initialize=list(self.data.generators.keys()))
        m.W = pyo.Set(initialize=list(self.data.wind_farms.keys()))
        m.B = pyo.Set(initialize=list(self.data.branches.keys()))
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
        
        def wind_forecast_init(m, w, t):
            return self.data.wind_farms[w]['forecast'][t]
        m.wind_forecast = pyo.Param(m.W, m.T, initialize=wind_forecast_init)
        
        def load_init(m, t):
            return self.data.loads[t]
        m.Pd = pyo.Param(m.T, initialize=load_init)
        
        def branch_capacity_init(m, b):
            return self.data.branches[b]['capacity']
        m.branch_capacity = pyo.Param(m.B, initialize=branch_capacity_init)
        
        # PTDF参数
        def ptdf_g_init(m, b, g):
            gen_bus = self.data.generators[g]['bus']
            return self.data.ptdf[b].get(f'G{gen_bus}', 0)
        m.ptdf_g = pyo.Param(m.B, m.G, initialize=ptdf_g_init)
        
        def ptdf_w_init(m, b, w):
            wind_bus = self.data.wind_farms[w]['bus']
            return self.data.ptdf[b].get(f'G{wind_bus}', 0)
        m.ptdf_w = pyo.Param(m.B, m.W, initialize=ptdf_w_init)
        
        def ptdf_load_init(m, b):
            return self.data.ptdf[b]['load']
        m.ptdf_load = pyo.Param(m.B, initialize=ptdf_load_init)
        
        # 风险惩罚系数
        m.theta_eens = pyo.Param(initialize=theta_eens)
        m.theta_wind = pyo.Param(initialize=theta_wind)
        m.theta_flow = pyo.Param(initialize=theta_flow)
        
        # CCV参数
        def eens_a_init(m, t, k):
            return self.data.eens_a[k-1, t-1]
        m.eens_a = pyo.Param(m.T, m.K, initialize=eens_a_init)
        
        def eens_b_init(m, t, k):
            return self.data.eens_b[k-1, t-1]
        m.eens_b = pyo.Param(m.T, m.K, initialize=eens_b_init)
        
        def wc_a_init(m, t, k):
            return self.data.wc_a[k-1, t-1]
        m.wc_a = pyo.Param(m.T, m.K, initialize=wc_a_init)
        
        def wc_b_init(m, t, k):
            return self.data.wc_b[k-1, t-1]
        m.wc_b = pyo.Param(m.T, m.K, initialize=wc_b_init)
        
        if include_flow_risk:
            def flow_a_init(m, t, k):
                return self.data.flow_a[k-1, t-1]
            m.flow_a = pyo.Param(m.T, m.K, initialize=flow_a_init)
            
            def flow_b_init(m, t, k):
                return self.data.flow_b[k-1, t-1]
            m.flow_b = pyo.Param(m.T, m.K, initialize=flow_b_init)
        
        # 变量
        m.u = pyo.Var(m.G, m.T, domain=pyo.Binary)  # 启停状态
        m.p = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)  # 出力
        m.r_up = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)  # 上备用
        m.r_down = pyo.Var(m.G, m.T, domain=pyo.NonNegativeReals)  # 下备用
        
        # 风电弃风变量
        m.p_wind_curtail = pyo.Var(m.W, m.T, domain=pyo.NonNegativeReals)
        
        # 启停指示变量
        m.startup = pyo.Var(m.G, m.T, domain=pyo.Binary)
        
        # 支路潮流变量
        m.f = pyo.Var(m.B, m.T, domain=pyo.Reals)
        
        # 风险变量
        m.risk_eens = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        m.risk_wc = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        if include_flow_risk:
            m.risk_flow = pyo.Var(m.T, domain=pyo.NonNegativeReals)
        
        # ========== 约束 ==========
        # 1. 功率平衡
        def power_balance_rule(m, t):
            # 总发电 + 风电 = 负荷
            total_gen = sum(m.p[g, t] for g in m.G)
            total_wind = sum(m.wind_forecast[w, t] - m.p_wind_curtail[w, t] for w in m.W)
            return total_gen + total_wind == m.Pd[t]
        m.power_balance = pyo.Constraint(m.T, rule=power_balance_rule)
        
        # 2. 风电弃风限制
        def wind_curtail_limit_rule(m, w, t):
            return m.p_wind_curtail[w, t] <= m.wind_forecast[w, t]
        m.wind_curtail_limit = pyo.Constraint(m.W, m.T, rule=wind_curtail_limit_rule)
        
        # 3. 发电机出力上下限
        def gen_limit_lower_rule(m, g, t):
            return m.p[g, t] >= m.Pmin[g] * m.u[g, t]
        
        def gen_limit_upper_rule(m, g, t):
            return m.p[g, t] <= m.Pmax[g] * m.u[g, t]
        
        m.gen_limit_lower = pyo.Constraint(m.G, m.T, rule=gen_limit_lower_rule)
        m.gen_limit_upper = pyo.Constraint(m.G, m.T, rule=gen_limit_upper_rule)
        
        # 4. 备用约束
        def reserve_up_limit_rule(m, g, t):
            return m.r_up[g, t] <= m.u[g, t] * m.Pmax[g] - m.p[g, t]
        
        def reserve_down_limit_rule(m, g, t):
            return m.r_down[g, t] <= m.p[g, t] - m.Pmin[g] * m.u[g, t]
        
        m.reserve_up_limit = pyo.Constraint(m.G, m.T, rule=reserve_up_limit_rule)
        m.reserve_down_limit = pyo.Constraint(m.G, m.T, rule=reserve_down_limit_rule)
        
        # 5. 启停逻辑
        def startup_rule1(m, g, t):
            if t == 1:
                return m.startup[g, t] >= m.u[g, t]  # 假设初始状态为0
            else:
                return m.startup[g, t] >= m.u[g, t] - m.u[g, t-1]
        
        def startup_rule2(m, g, t):
            return m.startup[g, t] <= m.u[g, t]
        
        def startup_rule3(m, g, t):
            if t == 1:
                return pyo.Constraint.Skip
            return m.startup[g, t] <= 1 - m.u[g, t-1]
        
        m.startup_rule1 = pyo.Constraint(m.G, m.T, rule=startup_rule1)
        m.startup_rule2 = pyo.Constraint(m.G, m.T, rule=startup_rule2)
        m.startup_rule3 = pyo.Constraint(m.G, m.T, rule=startup_rule3)
        
        # 6. 支路潮流计算
        def branch_flow_rule(m, b, t):
            # 发电机贡献
            gen_flow = sum(m.ptdf_g[b, g] * m.p[g, t] for g in m.G)
            
            # 风电贡献（实际风电 = 预测 - 弃风）
            wind_flow = sum(m.ptdf_w[b, w] * (m.wind_forecast[w, t] - m.p_wind_curtail[w, t]) for w in m.W)
            
            # 负荷贡献
            load_flow = -m.ptdf_load[b] * m.Pd[t]
            
            return m.f[b, t] == gen_flow + wind_flow + load_flow
        m.branch_flow = pyo.Constraint(m.B, m.T, rule=branch_flow_rule)
        
        # 7. 支路容量约束（硬约束）
        def branch_capacity_rule(m, b, t):
            return (-m.branch_capacity[b], m.f[b, t], m.branch_capacity[b])
        m.branch_capacity_constraint = pyo.Constraint(m.B, m.T, rule=branch_capacity_rule)
        
        # 8. CCV EENS风险约束
        def eens_risk_rule(m, t, k):
            # 总上备用
            total_up = sum(m.r_up[g, t] for g in m.G)
            # 风险 ≥ a * 备用 + b
            return m.risk_eens[t] >= m.eens_a[t, k] * total_up + m.eens_b[t, k]
        m.eens_risk_constraint = pyo.Constraint(m.T, m.K, rule=eens_risk_rule)
        
        # 9. CCV弃风风险约束
        def wc_risk_rule(m, t, k):
            # 总下备用
            total_down = sum(m.r_down[g, t] for g in m.G)
            return m.risk_wc[t] >= m.wc_a[t, k] * total_down + m.wc_b[t, k]
        m.wc_risk_constraint = pyo.Constraint(m.T, m.K, rule=wc_risk_rule)
        
        # 10. CCV支路溢出风险约束
        if include_flow_risk:
            def flow_risk_rule(m, t, k):
                # 总上备用
                total_up = sum(m.r_up[g, t] for g in m.G)
                return m.risk_flow[t] >= m.flow_a[t, k] * total_up + m.flow_b[t, k]
            m.flow_risk_constraint = pyo.Constraint(m.T, m.K, rule=flow_risk_rule)
        
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
            
            # 风险成本
            risk_cost = sum(m.theta_eens * m.risk_eens[t] for t in m.T)
            risk_cost += sum(m.theta_wind * m.risk_wc[t] for t in m.T)
            if include_flow_risk:
                risk_cost += sum(m.theta_flow * m.risk_flow[t] for t in m.T)
            
            return gen_cost + reserve_cost + startup_cost_total + risk_cost
        
        m.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        self.model = m
        return m
    
    def solve(self, solver_name='gurobi', tee=False):
        """求解模型"""
        if solver_name == 'gurobi':
            solver = SolverFactory('gurobi')
            solver.options['MIPGap'] = 0.001
            solver.options['TimeLimit'] = 300
        elif solver_name == 'cplex':
            solver = SolverFactory('cplex')
            solver.options['mipgap'] = 0.001
        else:
            solver = SolverFactory('glpk')
        
        results = solver.solve(self.model, tee=tee)
        
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            return True, results
        else:
            print(f"求解状态: {results.solver.termination_condition}")
            return False, results
    
    def get_results(self):
        """获取详细结果"""
        if self.model is None:
            return None
        
        m = self.model
        
        results = {
            'total_cost': pyo.value(m.objective),
            'generation': {},
            'wind_curtail': {},
            'reserve_up': {},
            'reserve_down': {},
            'branch_flow': {},
            'risk_eens': {},
            'risk_wc': {},
            'risk_flow': {},
            'status': {},
            'startup': {},
            'cost_breakdown': {}
        }
        
        # 发电量
        for g in m.G:
            results['generation'][g] = {}
            for t in m.T:
                results['generation'][g][t] = pyo.value(m.p[g, t])
        
        # 风电弃风
        for w in m.W:
            results['wind_curtail'][w] = {}
            for t in m.T:
                results['wind_curtail'][w][t] = pyo.value(m.p_wind_curtail[w, t])
        
        # 备用
        for g in m.G:
            results['reserve_up'][g] = {}
            results['reserve_down'][g] = {}
            for t in m.T:
                results['reserve_up'][g][t] = pyo.value(m.r_up[g, t])
                results['reserve_down'][g][t] = pyo.value(m.r_down[g, t])
        
        # 支路潮流
        for b in m.B:
            results['branch_flow'][b] = {}
            for t in m.T:
                results['branch_flow'][b][t] = pyo.value(m.f[b, t])
        
        # 风险
        for t in m.T:
            results['risk_eens'][t] = pyo.value(m.risk_eens[t])
            results['risk_wc'][t] = pyo.value(m.risk_wc[t])
            if hasattr(m, 'risk_flow'):
                results['risk_flow'][t] = pyo.value(m.risk_flow[t])
            else:
                results['risk_flow'][t] = 0.0
        
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
        risk_cost += sum(pyo.value(m.theta_wind) * pyo.value(m.risk_wc[t]) for t in m.T)
        if hasattr(m, 'risk_flow'):
            risk_cost += sum(pyo.value(m.theta_flow) * pyo.value(m.risk_flow[t]) for t in m.T)
        
        results['cost_breakdown'] = {
            'generation': gen_cost,
            'reserve': reserve_cost,
            'startup': startup_cost,
            'risk': risk_cost
        }
        
        return results

# ==================== 3. 敏感性分析器 ====================
class CompleteSensitivityAnalyzer:
    """进行完整敏感性分析"""
    
    def __init__(self, data):
        self.data = data
    
    def run_theta_sensitivity(self, theta_values, include_flow_risk=True, debug=False):
        """运行θ敏感性分析"""
        results = []
        
        for theta in theta_values:
            if debug:
                print(f"\n{'='*60}")
                print(f"Solving for θ_EENS = {theta}")
                print('='*60)
            
            # 构建模型
            builder = CompletePaperModel(self.data)
            
            try:
                model = builder.build_model(
                    theta_eens=theta, 
                    theta_wind=theta,  # 与θ_eens相同
                    theta_flow=theta,  # 与θ_eens相同
                    include_flow_risk=include_flow_risk
                )
                
                # 设置初始值
                for g in model.G:
                    for t in model.T:
                        model.u[g, t].value = 1 if g == 'G1' or g == 'G2' else 0
                        model.p[g, t].value = 10.0  # 最小出力
                
                # 求解
                success, _ = builder.solve(solver_name='gurobi', tee=False)
                
                if success:
                    # 获取结果
                    model_results = builder.get_results()
                    
                    # 关键指标
                    g2_output_hour3 = model_results['generation']['G2'][3]
                    g1_output_hour3 = model_results['generation']['G1'][3]
                    g3_output_hour3 = model_results['generation']['G3'][3]
                    total_cost = model_results['total_cost']
                    
                    # 计算总风险
                    total_eens_risk = sum(model_results['risk_eens'][t] for t in self.data.T)
                    total_wc_risk = sum(model_results['risk_wc'][t] for t in self.data.T)
                    total_flow_risk = sum(model_results['risk_flow'][t] for t in self.data.T)
                    total_risk = total_eens_risk + total_wc_risk + total_flow_risk
                    
                    # 计算总备用
                    total_reserve_up = sum(
                        sum(model_results['reserve_up'][g][t] for g in ['G1', 'G2', 'G3']) 
                        for t in self.data.T
                    )
                    
                    # 计算支路潮流
                    branch3_flow_hour3 = model_results['branch_flow']['L3'][3]
                    
                    # 记录
                    result = {
                        'theta': theta,
                        'G2_hour3': g2_output_hour3,
                        'G1_hour3': g1_output_hour3,
                        'G3_hour3': g3_output_hour3,
                        'total_cost': total_cost,
                        'total_risk': total_risk,
                        'total_reserve_up': total_reserve_up,
                        'branch3_flow_hour3': branch3_flow_hour3,
                        'results': model_results
                    }
                    
                    results.append(result)
                    
                    if debug:
                        print(f"  G1 at hour 3: {g1_output_hour3:.2f} MW")
                        print(f"  G2 at hour 3: {g2_output_hour3:.2f} MW")
                        print(f"  G3 at hour 3: {g3_output_hour3:.2f} MW")
                        print(f"  Branch 3 flow: {branch3_flow_hour3:.2f} MW (limit: 55 MW)")
                        print(f"  Total cost: ${total_cost:.2f}")
                        print(f"  Total risk: {total_risk:.4f} MW")
                        print(f"  Total reserve up: {total_reserve_up:.2f} MW")
                else:
                    print(f"  Failed to solve for θ = {theta}")
                    results.append(None)
                    
            except Exception as e:
                print(f"  Error for θ = {theta}: {e}")
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
                'G1@H3': r['G1_hour3'],
                'G2@H3': r['G2_hour3'],
                'G3@H3': r['G3_hour3'],
                'Branch3': r['branch3_flow_hour3'],
                'Total Cost': r['total_cost'],
                'Total Risk': r['total_risk'],
                'Total Reserve': r['total_reserve_up']
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # 分析G2变化趋势
        if len(valid_results) > 1:
            min_theta = valid_results[0]['theta']
            max_theta = valid_results[-1]['theta']
            g2_change = valid_results[-1]['G2_hour3'] - valid_results[0]['G2_hour3']
            
            print(f"\nG2变化分析:")
            print(f"  θ从 {min_theta} 到 {max_theta}")
            print(f"  G2出力变化: {g2_change:.2f} MW")
            
            if abs(g2_change) > 0.1:
                if g2_change > 0:
                    print("  ✓ G2出力随θ增加而增加 (符合Fig.4趋势)")
                else:
                    print("  ✗ G2出力随θ增加而减少")
            else:
                print("  ⚠ G2出力基本不变")
            
            # 分析风险变化
            risk_change = valid_results[-1]['total_risk'] - valid_results[0]['total_risk']
            print(f"\n风险变化分析:")
            print(f"  总风险变化: {risk_change:.4f} MW")
            
            if risk_change < 0:
                print("  ✓ 风险随θ增加而减少 (符合预期)")
            else:
                print("  ✗ 风险没有减少")
        
        # 与Table II对比
        print("\n与论文Table II对比 (第3小时):")
        print("  论文中: G1=12, G2=74, G3=17, 总计=103 MW")
        
        # 找到最接近的θ值
        best_match = None
        best_diff = float('inf')
        
        for r in valid_results:
            total_gen = r['G1_hour3'] + r['G2_hour3'] + r['G3_hour3']
            diff = abs(total_gen - 103)
            if diff < best_diff:
                best_diff = diff
                best_match = r
        
        if best_match:
            print(f"\n最接近的结果 (θ={best_match['theta']}):")
            print(f"  G1: {best_match['G1_hour3']:.2f} MW")
            print(f"  G2: {best_match['G2_hour3']:.2f} MW")
            print(f"  G3: {best_match['G3_hour3']:.2f} MW")
            print(f"  总计: {best_match['G1_hour3']+best_match['G2_hour3']+best_match['G3_hour3']:.2f} MW")
            print(f"  支路3潮流: {best_match['branch3_flow_hour3']:.2f} MW")
            print(f"  与论文差异: {best_diff:.2f} MW")

# ==================== 4. 可视化 ====================
class CompleteVisualizer:
    """绘制完整图形"""
    
    @staticmethod
    def plot_figure4(ax, results):
        """绘制Fig.4: G2出力 vs θ"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            ax.text(0.5, 0.5, 'No valid results', ha='center', va='center')
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
            if i == 0 or i == len(theta_values)-1 or i % 3 == 0:
                ax.annotate(f'{g2:.1f}', (theta, g2), 
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', va='bottom', fontsize=9)
    
    @staticmethod
    def plot_figure5(ax, results):
        """绘制Fig.5: 总成本 vs θ"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            ax.text(0.5, 0.5, 'No valid results', ha='center', va='center')
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
            ax.text(0.5, 0.5, 'No valid results', ha='center', va='center')
            return
        
        theta_values = [r['theta'] for r in valid_results]
        total_risks = [r['total_risk'] for r in valid_results]
        
        ax.plot(theta_values, total_risks, 'g-', linewidth=2, marker='^', markersize=6)
        ax.set_xlabel('Risk Penalty Coefficient θ_EENS ($/MWh)', fontsize=12)
        ax.set_ylabel('Total Risk (MW)', fontsize=12)
        ax.set_title('Fig.7: System Risk vs Risk Penalty', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(theta_values))
        
        # 如果风险值差异大，使用对数坐标
        if max(total_risks) > 10 * min(total_risks) and min(total_risks) > 0:
            ax.set_yscale('log')
    
    @staticmethod
    def plot_g2_all_hours(ax, results):
        """绘制G2在所有时段的出力"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results or len(valid_results) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            return
        
        # 选择几个关键θ值
        selected_indices = [0, len(valid_results)//2, -1]
        selected_results = [valid_results[i] for i in selected_indices]
        theta_labels = [f'θ={valid_results[i]["theta"]}' for i in selected_indices]
        
        # 获取G2在所有时段的出力
        hours = [1, 2, 3, 4]
        g2_outputs = []
        
        for r in selected_results:
            outputs = []
            for t in hours:
                outputs.append(r['results']['generation']['G2'][t])
            g2_outputs.append(outputs)
        
        # 绘制柱状图
        x = np.arange(len(hours))
        width = 0.25
        
        for i, outputs in enumerate(g2_outputs):
            offset = (i - 1) * width
            ax.bar(x + offset, outputs, width, label=theta_labels[i])
        
        ax.set_xlabel('Hour', fontsize=12)
        ax.set_ylabel('G2 Output (MW)', fontsize=12)
        ax.set_title('G2 Output at Different Hours and θ Values', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Hour {h}' for h in hours])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

# ==================== 5. 调试函数 ====================
def debug_single_case_complete(theta=100):
    """调试单个θ值（完整模型）"""
    print("\n" + "="*80)
    print(f"详细调试: θ = {theta}")
    print("="*80)
    
    data = PaperDataPrecise()
    builder = CompletePaperModel(data)
    
    try:
        model = builder.build_model(
            theta_eens=theta, 
            theta_wind=theta,
            theta_flow=theta,
            include_flow_risk=True
        )
        
        # 设置初始值
        for g in model.G:
            for t in model.T:
                model.u[g, t].value = 1 if g == 'G1' or g == 'G2' else 0
                model.p[g, t].value = 10.0
        
        # 求解
        success, _ = builder.solve(solver_name='gurobi', tee=False)
        
        if success:
            results = builder.get_results()
            
            print(f"\n总成本: ${results['total_cost']:.2f}")
            print("\n成本分解:")
            for cost_type, value in results['cost_breakdown'].items():
                print(f"  {cost_type}: ${value:.2f}")
            
            print("\n第3小时详细结果:")
            t = 3
            print(f"  负荷: {data.loads[t]:.2f} MW")
            print(f"  风电预测: {sum(data.wind_farms[w]['forecast'][t] for w in data.wind_farms):.2f} MW")
            
            # 发电量
            print(f"\n  发电量 (MW):")
            total_gen = 0
            for g in ['G1', 'G2', 'G3']:
                gen = results['generation'][g][t]
                total_gen += gen
                print(f"    {g}: {gen:.2f}")
            print(f"    总计: {total_gen:.2f}")
            
            # 风电弃风
            print(f"\n  风电弃风 (MW):")
            for w in ['W1', 'W2']:
                curtail = results['wind_curtail'][w][t]
                print(f"    {w}: {curtail:.2f}")
            
            # 备用
            print(f"\n  上备用 (MW):")
            total_up = 0
            for g in ['G1', 'G2', 'G3']:
                up = results['reserve_up'][g][t]
                total_up += up
                print(f"    {g}: {up:.2f}")
            print(f"    总计: {total_up:.2f}")
            
            # 支路潮流
            print(f"\n  支路潮流 (MW):")
            for b in ['L1', 'L2', 'L3']:
                flow = results['branch_flow'][b][t]
                capacity = data.branches[b]['capacity']
                print(f"    {b}: {flow:.2f} (limit: ±{capacity})")
            
            # 风险
            print(f"\n  风险值:")
            print(f"    EENS风险: {results['risk_eens'][t]:.6f} MW")
            print(f"    弃风风险: {results['risk_wc'][t]:.6f} MW")
            print(f"    支路溢出风险: {results['risk_flow'][t]:.6f} MW")
            
            # 检查网络约束
            print(f"\n  网络约束检查:")
            for b in ['L1', 'L2', 'L3']:
                flow = results['branch_flow'][b][t]
                capacity = data.branches[b]['capacity']
                if abs(flow) > capacity:
                    print(f"    {b}: 超出限制! {abs(flow):.2f} > {capacity}")
                else:
                    margin = capacity - abs(flow)
                    print(f"    {b}: 安全裕度 {margin:.2f} MW")
            
            # 边际成本分析
            print(f"\n  边际成本分析:")
            for g in ['G1', 'G2', 'G3']:
                if results['generation'][g][t] > 0:
                    marginal_cost = data.generators[g]['cost_b']
                    print(f"    {g}: 边际成本 = ${marginal_cost:.2f}/MW")
            
            return results
        else:
            print("求解失败")
            return None
            
    except Exception as e:
        print(f"构建/求解模型时出错: {e}")
        return None

# ==================== 6. 主程序 ====================
def main_complete():
    """主程序（完整模型）"""
    print("\n" + "="*80)
    print("IEEE TPWRS 2015 - 完整复现程序（含网络约束）")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 加载论文数据（含网络数据）...")
    data = PaperDataPrecise()
    
    # 2. 调试单个案例
    print("\n2. 调试基准案例 (θ=100)...")
    debug_results = debug_single_case_complete(theta=100)
    
    # 3. 运行敏感性分析
    print("\n3. 运行敏感性分析...")
    analyzer = CompleteSensitivityAnalyzer(data)
    
    # θ值范围（论文范围）
    theta_values = [0, 10, 20, 50, 100, 200, 300, 400, 500]
    
    results = analyzer.run_theta_sensitivity(
        theta_values=theta_values,
        include_flow_risk=True,
        debug=True
    )
    
    # 4. 分析趋势
    analyzer.analyze_trends(results)
    
    # 5. 绘制图形
    print("\n4. 绘制论文图形...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    CompleteVisualizer.plot_figure4(axes[0], results)
    CompleteVisualizer.plot_figure5(axes[1], results)
    CompleteVisualizer.plot_figure7(axes[2], results)
    CompleteVisualizer.plot_g2_all_hours(axes[3], results)
    
    plt.tight_layout()
    plt.savefig('complete_replication.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 导出详细结果
    print("\n5. 导出详细结果...")
    export_detailed_results(results)
    
    print("\n" + "="*80)
    print("完整复现完成!")
    print("="*80)

def export_detailed_results(results):
    """导出详细结果到CSV"""
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("没有有效结果可以导出")
        return
    
    # 创建DataFrame
    data = []
    for r in valid_results:
        row = {
            'theta': r['theta'],
            'G1_hour3': r['G1_hour3'],
            'G2_hour3': r['G2_hour3'],
            'G3_hour3': r['G3_hour3'],
            'branch3_flow': r['branch3_flow_hour3'],
            'total_cost': r['total_cost'],
            'total_risk': r['total_risk'],
            'total_reserve_up': r['total_reserve_up']
        }
        
        # 添加所有时段的G2出力
        for t in [1, 2, 3, 4]:
            row[f'G2_hour{t}'] = r['results']['generation']['G2'][t]
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv('complete_replication_results.csv', index=False)
    print(f"结果已导出到 complete_replication_results.csv ({len(df)} 行)")
    
    # 显示关键结果
    print("\n关键结果摘要:")
    print(df[['theta', 'G2_hour1', 'G2_hour2', 'G2_hour3', 'G2_hour4', 'total_cost']].to_string(index=False))

# ==================== 7. 简化版本（快速测试） ====================
def quick_test():
    """快速测试版本"""
    print("\n" + "="*80)
    print("快速测试 - 检查关键问题")
    print("="*80)
    
    data = PaperDataPrecise()
    
    # 测试两个极端情况
    print("\n测试θ=0（无风险惩罚）:")
    builder1 = CompletePaperModel(data)
    model1 = builder1.build_model(theta_eens=0, include_flow_risk=False)
    success1, _ = builder1.solve(solver_name='gurobi', tee=False)
    
    if success1:
        results1 = builder1.get_results()
        g2_0 = results1['generation']['G2'][3]
        print(f"  G2 at hour 3: {g2_0:.2f} MW")
        print(f"  Branch 3 flow: {results1['branch_flow']['L3'][3]:.2f} MW")
    else:
        print("  求解失败")
    
    print("\n测试θ=500（高风险惩罚）:")
    builder2 = CompletePaperModel(data)
    model2 = builder2.build_model(theta_eens=500, include_flow_risk=False)
    success2, _ = builder2.solve(solver_name='gurobi', tee=False)
    
    if success2:
        results2 = builder2.get_results()
        g2_500 = results2['generation']['G2'][3]
        print(f"  G2 at hour 3: {g2_500:.2f} MW")
        print(f"  Branch 3 flow: {results2['branch_flow']['L3'][3]:.2f} MW")
        
        if abs(g2_500 - g2_0) > 0.1:
            print(f"\n✓ G2出力变化: {g2_500 - g2_0:.2f} MW")
        else:
            print(f"\n⚠ G2出力基本不变")
    else:
        print("  求解失败")

# ==================== 执行 ====================
if __name__ == "__main__":
    # 导入依赖检查
    try:
        import pyomo.environ as pyo
    except ImportError:
        print("请安装所需依赖:")
        print("pip install numpy matplotlib pandas pyomo")
        exit(1)
    
    # 选择运行模式
    print("选择运行模式:")
    print("1. 完整复现（含网络约束）")
    print("2. 快速测试")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == '1':
        main_complete()
    elif choice == '2':
        quick_test()
    else:
        print("无效选择，运行完整复现")
        main_complete()