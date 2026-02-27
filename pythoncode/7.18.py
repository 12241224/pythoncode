import pyomo.environ as pyo
from pyomo.environ import *
import numpy as np

# ========== 输入部分 ==========
# 1. 加载负荷曲线数据（单位：100MW）
load_factor = {
    0: 0.756129, 1: 0.726395, 2: 0.704225, 3: 0.696922, 4: 0.696922,
    5: 0.704225, 6: 0.807773, 7: 0.896453, 8: 0.962963, 9: 0.970527,
    10: 0.970527, 11: 0.962963, 12: 0.962963, 13: 0.962963, 14: 0.948357,
    15: 0.95566, 16: 0.991393, 17: 1.0, 18: 1.0, 19: 0.970527,
    20: 0.933229, 21: 0.874283, 22: 0.800469, 23: 0.726395
}
base_load = 100  # 基准负荷为100MW
hourly_load = {t: load_factor[t] * base_load for t in range(24)}  # 实际负荷（MW）

# 2. 机组参数（单位转换：100MW -> MW）
generators = {
    'G1': {
        'bus': 'Bus_1',
        'Pmax': 1.5749 * 100,       # 装机容量 -> 157.49 MW
        'Pmin': 0.5 * 100,          # 最小技术出力 -> 50 MW
        'min_up_time': 2,           # 最短开机时间（小时）
        'min_down_time': 2,         # 最短关机时间（小时）
        'ramp_up': 0.375 * 100,     # 上爬坡速率 -> 37.5 MW/h
        'ramp_down': 0.375 * 100,   # 下爬坡速率 -> 37.5 MW/h
        'startup_cost': 39373,      # 启动费用（元）
        'shutdown_cost': 19886.25,  # 停机费用（元）
        'a': 15.24,                 # 成本系数A（元/(100MW)^2）
        'b': 3853.97,               # 成本系数B（元/100MW）
        'c': 78679.88               # 成本系数C（元）
    },
    'G2': {
        'bus': 'Bus_2',
        'Pmax': 1.0 * 100,
        'Pmin': 0.25 * 100,
        'min_up_time': 2,
        'min_down_time': 2,
        'ramp_up': 0.3 * 100,
        'ramp_down': 0.3 * 100,
        'startup_cost': 25000,
        'shutdown_cost': 12500,
        'a': 10.58,
        'b': 4615.91,
        'c': 94563.32
    },
    'G3': {
        'bus': 'Bus_5',
        'Pmax': 0.6 * 100,
        'Pmin': 0.15 * 100,
        'min_up_time': 2,
        'min_down_time': 2,
        'ramp_up': 0.15 * 100,
        'ramp_down': 0.15 * 100,
        'startup_cost': 15000,
        'shutdown_cost': 7500,
        'a': 2.8,
        'b': 4039.65,
        'c': 104999.8
    },
    'G4': {
        'bus': 'Bus_8',
        'Pmax': 0.8 * 100,
        'Pmin': 0.2 * 100,
        'min_up_time': 2,
        'min_down_time': 2,
        'ramp_up': 0.2 * 100,
        'ramp_down': 0.2 * 100,
        'startup_cost': 20000,
        'shutdown_cost': 10000,
        'a': 3.54,
        'b': 3830.55,
        'c': 124353.1
    },
    'G5': {
        'bus': 'Bus_11',
        'Pmax': 0.4 * 100,
        'Pmin': 0.1 * 100,
        'min_up_time': 2,
        'min_down_time': 2,
        'ramp_up': 0.15 * 100,
        'ramp_down': 0.15 * 100,
        'startup_cost': 10000,
        'shutdown_cost': 5000,
        'a': 2.11,
        'b': 3632.78,
        'c': 165857
    },
    'G6': {
        'bus': 'Bus_13',
        'Pmax': 0.4 * 100,
        'Pmin': 0.1 * 100,
        'min_up_time': 2,
        'min_down_time': 2,
        'ramp_up': 0.15 * 100,
        'ramp_down': 0.15 * 100,
        'startup_cost': 10000,
        'shutdown_cost': 5000,
        'a': 1.79,
        'b': 3827.04,
        'c': 135665.9
    },
}

# ========== 建模部分 ==========
model = pyo.ConcreteModel()

# 定义集合
model.G = pyo.Set(initialize=generators.keys())  # 机组集合
model.T = pyo.Set(initialize=range(24))          # 时间集合

# 定义变量
model.u = pyo.Var(model.G, model.T, within=pyo.Binary)     # 机组启停状态
model.p = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)  # 机组出力（MW）
model.su = pyo.Var(model.G, model.T, within=pyo.Binary)    # 启动标志
model.sd = pyo.Var(model.G, model.T, within=pyo.Binary)    # 停机标志

# 辅助变量（单位转换为100MW，用于成本计算）
model.p_100mw = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)  # 机组出力（100MW）

# 目标函数：总成本 = 运行成本 + 启停成本
def objective_rule(model):
    total_cost = 0
    for g in model.G:
        for t in model.T:
            # 运行成本（直接使用二次表达式）
            total_cost += (generators[g]['a'] * model.p_100mw[g, t]**2 + 
                          generators[g]['b'] * model.p_100mw[g, t] + 
                          generators[g]['c'])
            # 启停成本
            total_cost += generators[g]['startup_cost'] * model.su[g, t]
            total_cost += generators[g]['shutdown_cost'] * model.sd[g, t]
    return total_cost
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# 1. 机组出力上下限约束
def power_lower_limit_rule(model, g, t):
    return model.p[g, t] >= generators[g]['Pmin'] * model.u[g, t]

def power_upper_limit_rule(model, g, t):
    return model.p[g, t] <= generators[g]['Pmax'] * model.u[g, t]

model.power_lower_limit = pyo.Constraint(model.G, model.T, rule=power_lower_limit_rule)
model.power_upper_limit = pyo.Constraint(model.G, model.T, rule=power_upper_limit_rule)

# 2. 单位转换约束
def unit_conversion_rule(model, g, t):
    return model.p_100mw[g, t] == model.p[g, t] / 100.0

model.unit_conversion = pyo.Constraint(model.G, model.T, rule=unit_conversion_rule)

# 3. 启停逻辑约束
def startup_rule(model, g, t):
    if t == 0: 
        # 初始状态假设为停机，实际应用需根据具体情况设置
        return pyo.Constraint.Skip
    return model.su[g, t] >= model.u[g, t] - model.u[g, t-1]

def shutdown_rule(model, g, t):
    if t == 0: 
        return pyo.Constraint.Skip
    return model.sd[g, t] >= model.u[g, t-1] - model.u[g, t]

model.startup_constr = pyo.Constraint(model.G, model.T, rule=startup_rule)
model.shutdown_constr = pyo.Constraint(model.G, model.T, rule=shutdown_rule)

# 4. 总功率平衡约束
def power_balance_rule(model, t):
    total_gen = sum(model.p[g, t] for g in model.G)
    return total_gen == hourly_load[t]

model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

# 5. 爬坡约束
def ramp_up_rule(model, g, t):
    if t == 0: 
        # 初始出力设为0，实际应用需根据具体情况设置
        return pyo.Constraint.Skip
    return model.p[g, t] - model.p[g, t-1] <= generators[g]['ramp_up']

def ramp_down_rule(model, g, t):
    if t == 0: 
        return pyo.Constraint.Skip
    return model.p[g, t-1] - model.p[g, t] <= generators[g]['ramp_down']

model.ramp_up = pyo.Constraint(model.G, model.T, rule=ramp_up_rule)
model.ramp_down = pyo.Constraint(model.G, model.T, rule=ramp_down_rule)

# 6. 最小启停时间约束
def min_up_time_rule(model, g, t):
    min_up = generators[g]['min_up_time']
    if t < min_up: 
        return pyo.Constraint.Skip
    return sum(1 - model.u[g, k] for k in range(t - min_up + 1, t + 1)) <= min_up * (1 - model.su[g, t])

def min_down_time_rule(model, g, t):
    min_down = generators[g]['min_down_time']
    if t < min_down: 
        return pyo.Constraint.Skip
    return sum(model.u[g, k] for k in range(t - min_down + 1, t + 1)) <= min_down * (1 - model.sd[g, t])

model.min_up_time = pyo.Constraint(model.G, model.T, rule=min_up_time_rule)
model.min_down_time = pyo.Constraint(model.G, model.T, rule=min_down_time_rule)

# ========== 使用Gurobi求解 ==========
solver = pyo.SolverFactory('gurobi')  # 使用Gurobi求解器

# 设置求解器参数（可选）
solver.options['MIPGap'] = 0.001      # 设置MIP间隙容差
solver.options['TimeLimit'] = 300      # 设置求解时间限制（秒）

# 求解模型
results = solver.solve(model, tee=True)  # tee=True显示求解过程

# 检查求解状态
if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
    print("求解成功！")
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("模型不可行！")
else:
    print("求解未完成！状态:", results.solver.status)

# 打印结果
print("\n========== 机组启停计划 ==========")
for g in model.G:
    print(f"\n机组 {g}:")
    for t in model.T:
        status = "运行" if pyo.value(model.u[g, t]) > 0.5 else "停机"
        print(f"  时段 {t:2d}: 状态={status}, 出力={pyo.value(model.p[g, t]):.2f} MW")

print("\n========== 总运行成本 ==========")
print(f"{pyo.value(model.obj):.2f} 元")