import pyomo.environ as pyo
from pyomo.environ import *

import numpy as np
import pandas as pd

# 读取数据
all_sheets = pd.read_excel("C:\\Users\\admin\\Desktop\\3用这个算例-XJTU-ROTS.xlsx", sheet_name=None)

# 提取特定的 Sheet
thermal_units = all_sheets["UnitThermalGenerators"]  # 火电机组参数
load_profile = all_sheets["LoadCurves"]  # 负荷曲线
# 假设 load_profile 的第一列是时间，第二列是负荷数据
hourly_load = {t: load_profile.iloc[t, 1] for t in range(24)}  # 实际负荷（MW）
dict_data = thermal_units.to_dict(orient="dict")
# 创建模型
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=range(24))  # 24小时
model.G = pyo.Set(initialize=dict_data.keys())  # 机组集合

# 定义变量
model.u = pyo.Var(model.G, model.T, within=pyo.Binary)  # 启停状态
model.p = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)  # 出力
model.su = pyo.Var(model.G, model.T, within=pyo.Binary)    # 启动标志
model.sd = pyo.Var(model.G, model.T, within=pyo.Binary)    # 停机标志

# 分段线性化参数
N = 10  # 分段数
P_max = 600  # 最大出力为 600
delta = P_max / N  # 每个区间的宽度

# 引入分段变量
model.p_segments = pyo.Var(model.G, model.T, range(N), within=pyo.NonNegativeReals)
model.z_segments = pyo.Var(model.G, model.T, range(N), within=pyo.Binary)

# 约束：p[g, t] 的线性化
def p_linearization1(model, g, t):
    return sum(model.p_segments[g, t, i] for i in range(N)) == model.p[g, t]
model.p_linearization1 = pyo.Constraint(model.G, model.T, rule=p_linearization1)

def p_linearization2(model, g, t):
    return sum(model.p_segments[g, t, i] for i in range(N)) <= P_max
model.p_linearization2 = pyo.Constraint(model.G, model.T, rule=p_linearization2)

def p_linearization3(model, g, t, i):
    return model.p_segments[g, t, i] <= (i + 1) * delta * model.z_segments[g, t, i]
model.p_linearization3 = pyo.Constraint(model.G, model.T, range(N), rule=p_linearization3)

def p_linearization4(model, g, t, i):
    return model.p_segments[g, t, i] >= i * delta * model.z_segments[g, t, i]
model.p_linearization4 = pyo.Constraint(model.G, model.T, range(N), rule=p_linearization4)

def p_linearization5(model, g, t):
    return sum(model.z_segments[g, t, i] for i in range(N)) == 1
model.p_linearization5 = pyo.Constraint(model.G, model.T, rule=p_linearization5)

# 引入二次项的线性化变量
model.q = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)

# 约束：q[g, t] 的线性化
def q_linearization1(model, g, t):
    return model.q[g, t] == sum((i * delta + delta / 2) * model.p_segments[g, t, i] for i in range(N))
model.q_linearization1 = pyo.Constraint(model.G, model.T, rule=q_linearization1)

# 目标函数（线性化）
def objective_rule(model):
    cost = 0
    for g in model.G:
        startup_cost = thermal_units.loc[g, "启动费用(吨标准煤)"]
        for t in model.T:
            # 启停成本
            if t > 0:
                cost += startup_cost * (model.u[g, t] - model.u[g, t - 1])  # 简化逻辑
            else:
                initial_status = thermal_units.loc[g, "初始开机状态(TRUE/FALSE)"]
                cost += startup_cost * (model.u[g, t] - initial_status)
            # 运行成本（线性化）
            A = thermal_units.loc[g, "运行成本曲线系数A(吨标准煤/(MW)^2)"]
            B = thermal_units.loc[g, "运行成本曲线系数B(吨标准煤/MW)"]
            fuel_price = thermal_units.loc[g, "fuel_price"]
            cost += (A * model.q[g, t] + B * model.p[g, t]) * fuel_price
    return cost
model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# 约束 1. 机组出力上下限约束
def power_lower_limit_rule(model, g, t):
    return model.p[g, t] >= thermal_units.loc[g, "最小技术出力(MW)"] * model.u[g, t]

def power_upper_limit_rule(model, g, t):
    return model.p[g, t] <= thermal_units.loc[g, "装机容量(MW)"] * model.u[g, t]

model.power_lower_limit = pyo.Constraint(model.G, model.T, rule=power_lower_limit_rule)
model.power_upper_limit = pyo.Constraint(model.G, model.T, rule=power_upper_limit_rule)

# 约束 2: 启停逻辑
def startup_rule(model, g, t):
    if t == 0: return pyo.Constraint.Skip
    return model.su[g, t] >= model.u[g, t] - model.u[g, t-1]

def shutdown_rule(model, g, t):
    if t == 0: return pyo.Constraint.Skip
    return model.sd[g, t] >= model.u[g, t-1] - model.u[g, t]

model.startup_constr = pyo.Constraint(model.G, model.T, rule=startup_rule)
model.shutdown_constr = pyo.Constraint(model.G, model.T, rule=shutdown_rule)

# 约束 3: 总功率平衡
def power_balance_rule(model, t):
    total_gen = sum(model.p[g, t] for g in model.G)
    return total_gen == hourly_load [t]

model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

# 约束 4: 爬坡约束
def ramp_up_rule(model, g, t):
    if t == 0: return pyo.Constraint.Skip
    return model.p[g, t] - model.p[g, t-1] <= thermal_units.loc[g, "上爬坡速率(MW/h)"]

def ramp_down_rule(model, g, t):
    if t == 0: return pyo.Constraint.Skip
    return model.p[g, t-1] - model.p[g, t] <= thermal_units.loc[g, "下爬坡速率(MW/h)"]

model.ramp_up = pyo.Constraint(model.G, model.T, rule=ramp_up_rule)
model.ramp_down = pyo.Constraint(model.G, model.T, rule=ramp_down_rule)

# 约束 5: 最小启停时间约束
def min_up_time_rule(model, g, t):
    min_up = thermal_units.loc[g, "最短开机时间(h)"]
    if t < min_up: return pyo.Constraint.Skip
    return sum(1 - model.u[g, k] for k in range(t - min_up + 1, t + 1)) <= min_up * (1 - model.su[g, t])

def min_down_time_rule(model, g, t):
    min_down = thermal_units.loc[g, "最短关机时间(h)"]
    if t < min_down: return pyo.Constraint.Skip
    return sum(model.u[g, k] for k in range(t - min_down + 1, t + 1)) <= min_down * (1 - model.sd[g, t])

model.min_up_time = pyo.Constraint(model.G, model.T, rule=min_up_time_rule)
model.min_down_time = pyo.Constraint(model.G, model.T, rule=min_down_time_rule)


# 求解模型
solver = pyo.SolverFactory('cbc')  # 使用CBC求解器
results = solver.solve(model)

# 输出结果
for g in model.G:
    for t in model.T:
        print(f"机组{g}在时段{t}: 状态{model.u[g, t].value}, 出力{model.p[g, t].value}")









