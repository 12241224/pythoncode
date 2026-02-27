import pyomo.environ as pyo
from pyomo.environ import *
import numpy as np

# ========== 输入部分 ==========
# 机组参数（根据IEEE30系统假设）
generators = {
    'G1': {'bus': 'Bus_1', 'Pmin': 50, 'Pmax': 200, 'a': 0.003, 'b': 3.0, 'startup_cost': 200, 'shutdown_cost': 100},
    'G2': {'bus': 'Bus_2', 'Pmin': 20, 'Pmax': 80, 'a': 0.002, 'b': 3.5, 'startup_cost': 150, 'shutdown_cost': 80},
    'G3': {'bus': 'Bus_5', 'Pmin': 15, 'Pmax': 50, 'a': 0.005, 'b': 4.0, 'startup_cost': 100, 'shutdown_cost': 50},
    'G4': {'bus': 'Bus_8', 'Pmin': 10, 'Pmax': 35, 'a': 0.008, 'b': 5.0, 'startup_cost': 80, 'shutdown_cost': 40},
    'G5': {'bus': 'Bus_11', 'Pmin': 30, 'Pmax': 100, 'a': 0.004, 'b': 3.2, 'startup_cost': 180, 'shutdown_cost': 90},
    'G6': {'bus': 'Bus_13', 'Pmin': 40, 'Pmax': 150, 'a': 0.0035, 'b': 3.8, 'startup_cost': 220, 'shutdown_cost': 110},
}

# 节点负荷比例系数（从PDF第三页提取）
load_profiles = {
    'Bus_2': 0.217, 'Bus_3': 0.024, 'Bus_4': 0.076, 'Bus_5': 0.942,
    'Bus_7': 0.228, 'Bus_8': 0.3, 'Bus_10': 0.058, 'Bus_12': 0.112,
    'Bus_14': 0.062, 'Bus_15': 0.082, 'Bus_16': 0.035, 'Bus_17': 0.09,
    'Bus_18': 0.032, 'Bus_19': 0.095, 'Bus_20': 0.022, 'Bus_21': 0.175,
    'Bus_23': 0.032, 'Bus_24': 0.087, 'Bus_26': 0.035, 'Bus_29': 0.024,
    'Bus_30': 0.106
}

# 基础负荷曲线（24小时，假设总负荷基准）
# 计算 base_load，确保使用 np.sin 并对结果进行平方
base_load = [100 + 50 * (np.sin(2 * np.pi * (t - 6) / 24))**2 for t in range(24)]

# 时间集合
hours = range(24)

# ========== 建模部分 ==========
model = pyo.ConcreteModel()

# 定义集合
model.G = pyo.Set(initialize=generators.keys())  # 机组集合
model.T = pyo.Set(initialize=hours)             # 时间集合

# 定义变量
model.u = pyo.Var(model.G, model.T, within=pyo.Binary)    # 机组启停状态
model.p = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)  # 机组出力
model.su = pyo.Var(model.G, model.T, within=pyo.Binary)   # 启动标志
model.sd = pyo.Var(model.G, model.T, within=pyo.Binary)   # 停机标志

# 目标函数：总成本最小
def objective_rule(model):
    cost = 0
    for g in model.G:
        for t in model.T:
            cost += (generators[g]['a'] * model.p[g,t]**2 + generators[g]['b'] * model.p[g,t])  # 煤耗成本
            cost += generators[g]['startup_cost'] * model.su[g,t]  # 启动成本
            cost += generators[g]['shutdown_cost'] * model.sd[g,t]  # 停机成本
    return cost
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# 1. 机组出力上下限约束
def power_lower_limit_rule(model, g, t):
    return model.p[g, t] >= generators[g]['Pmin'] * model.u[g, t]

def power_upper_limit_rule(model, g, t):
    return model.p[g, t] <= generators[g]['Pmax'] * model.u[g, t]

model.power_lower_limit = pyo.Constraint(model.G, model.T, rule=power_lower_limit_rule)
model.power_upper_limit = pyo.Constraint(model.G, model.T, rule=power_upper_limit_rule)

# 约束 2: 启停逻辑
def startup_rule(model, g, t):
    if t == 0: return pyo.Constraint.Skip  # 初始状态已知
    return model.su[g,t] >= model.u[g,t] - model.u[g,t-1]
model.startup_constr = pyo.Constraint(model.G, model.T, rule=startup_rule)

def shutdown_rule(model, g, t):
    if t == 0: return pyo.Constraint.Skip
    return model.sd[g,t] >= model.u[g,t-1] - model.u[g,t]
model.shutdown_constr = pyo.Constraint(model.G, model.T, rule=shutdown_rule)

# 约束 3: 系统总功率平衡
def power_balance_rule(model, t):
    total_gen = sum(model.p[g,t] for g in model.G)
    total_load = sum(load_profiles[b] * base_load[t] for b in load_profiles)
    return total_gen == total_load
model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

# ========== 求解及输出 ==========
solver = pyo.SolverFactory('glpk')  # 使用开源求解器GLPK
results = solver.solve(model)

# 输出结果
print("最优机组组合计划：")
for t in model.T:
    print(f"\n--- 时段 {t} ---")
    for g in model.G:
        if model.u[g,t].value > 0.5:  # 机组运行
            print(f"  {g}: 出力 {model.p[g,t].value:.1f} MW (状态: 运行)")
        else:
            print(f"  {g}: 出力 0.0 MW (状态: 停机)")

print("\n总运行成本:", model.obj())