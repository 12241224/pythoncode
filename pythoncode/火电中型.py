import pyomo.environ as pyo
from pyomo.environ import *
import numpy as np
import pandas as pd

# ========== 输入部分 ==========
# 读取数据
all_sheets = pd.read_excel("C:\\Users\\admin\\Desktop\\3用这个算例-XJTU-ROTS.xlsx", sheet_name=None)

# 提取特定的 Sheet
thermal_units = all_sheets["UnitThermalGenerators"]  # 火电机组参数
load_profile = all_sheets["LoadCurves"]  # 负荷曲线
transmission_lines = all_sheets["NetLines"]  # 读取输电线路数据

# 确保最短开机时间和最短关机时间是整数
thermal_units["最短开机时间(h)"] = thermal_units["最短开机时间(h)"].astype(int)
thermal_units["最短关机时间(h)"] = thermal_units["最短关机时间(h)"].astype(int)

# 确保初始开机状态是整数（0 或 1）
thermal_units["初始开机状态(TRUE/FALSE)"] = thermal_units["初始开机状态(TRUE/FALSE)"].astype(int)

# 提取实际负荷
hourly_load = {t: load_profile.iloc[t, 1] * load_profile.iloc[t, 2] for t in range(24) if not pd.isna(load_profile.iloc[t, 1])}

# ========== 建模部分 ==========
# 创建模型
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=range(24))  # 24小时
model.G = pyo.Set(initialize=thermal_units.index)  # 机组集合
model.Lines = pyo.Set(initialize=transmission_lines.index)

# 定义变量
model.u = pyo.Var(model.G, model.T, within=pyo.Binary)  # 启停状态
model.p = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)  # 出力
model.su = pyo.Var(model.G, model.T, within=pyo.Binary)  # 启动标志
model.sd = pyo.Var(model.G, model.T, within=pyo.Binary)  # 停机标志
model.line_flow = pyo.Var(model.Lines, model.T, within=pyo.Reals)  # 定义线路功率变量（假设为双向）

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

# 目标函数（修正后的系数单位）
def objective_rule(model):
    cost = 0
    for g in model.G:
        startup_cost = thermal_units.loc[g, "启动费用(吨标准煤)"]  # 假设单位为吨标准煤
        A = thermal_units.loc[g, "运行成本曲线系数A(吨标准煤/(MW)^2)"] * 1e-4 * 1500  # 调整量纲
        B = thermal_units.loc[g, "运行成本曲线系数B(吨标准煤/MW)"] * 1e-2 * 1500
        C_val = thermal_units.loc[g, "运行成本曲线系数C(吨标准煤)"] * 1500
        for t in model.T:
            cost += startup_cost * model.su[g, t]
            cost += (A * model.q[g, t] + B * model.p[g, t] + C_val * model.u[g, t])
    return cost

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# 约束 1. 机组出力上下限约束
def power_upper_limit_rule(model, g, t):
    return model.p[g, t] <= thermal_units.loc[g, "装机容量(MW)"] * model.u[g, t]

model.power_upper_limit = pyo.Constraint(model.G, model.T, rule=power_upper_limit_rule)

# 约束 2: 启停逻辑
def startup_rule(model, g, t):
    if t == 0:
        return pyo.Constraint.Skip
    return model.su[g, t] >= model.u[g, t] - model.u[g, t-1]

def shutdown_rule(model, g, t):
    if t == 0:
        return pyo.Constraint.Skip
    return model.sd[g, t] >= model.u[g, t-1] - model.u[g, t]

model.startup_constr = pyo.Constraint(model.G, model.T, rule=startup_rule)
model.shutdown_constr = pyo.Constraint(model.G, model.T, rule=shutdown_rule)

# 约束 3: 总功率平衡
def power_balance_rule(model, t):
    total_gen = sum(model.p[g, t] for g in model.G)
    return total_gen == hourly_load[t]

model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

# 约束 4: 爬坡约束
def ramp_up_rule(model, g, t):
    if t == 0:
        return pyo.Constraint.Skip
    return model.p[g, t] - model.p[g, t-1] <= thermal_units.loc[g, "上爬坡速率(MW/h)"]

def ramp_down_rule(model, g, t):
    if t == 0:
        return pyo.Constraint.Skip
    return model.p[g, t-1] - model.p[g, t] <= thermal_units.loc[g, "下爬坡速率(MW/h)"]

model.ramp_up = pyo.Constraint(model.G, model.T, rule=ramp_up_rule)
model.ramp_down = pyo.Constraint(model.G, model.T, rule=ramp_down_rule)

# 约束 5: 最小启停时间约束
def min_up_time_rule(model, g, t):
    min_up = int(thermal_units.loc[g, "最短开机时间(h)"])  # 转换为整数
    if t < min_up:
        return pyo.Constraint.Skip
    return sum(1 - model.u[g, k] for k in range(t - min_up + 1, t + 1)) <= min_up * (1 - model.su[g, t])

def min_down_time_rule(model, g, t):
    min_down = int(thermal_units.loc[g, "最短关机时间(h)"])  # 转换为整数
    if t < min_down:
        return pyo.Constraint.Skip
    return sum(model.u[g, k] for k in range(t - min_down + 1, t + 1)) <= min_down * (1 - model.sd[g, t])

model.min_up_time = pyo.Constraint(model.G, model.T, rule=min_up_time_rule)
model.min_down_time = pyo.Constraint(model.G, model.T, rule=min_down_time_rule)

# 约束 6: 线路功率不超过容量
def line_capacity_rule(model, line, t):
    max_flow = transmission_lines.loc[line, "传输容量极限(MW)"]
    return (-max_flow, model.line_flow[line, t], max_flow)

model.line_capacity = pyo.Constraint(model.Lines, model.T, rule=line_capacity_rule)

#========求解模型=========
solver = pyo.SolverFactory('cbc')  # 使用CBC求解器
results = solver.solve(model)

# 输出求解结果
print("求解状态:", results.solver.status)
print("终止条件:", results.solver.termination_condition)

# 检查求解结果
if results.solver.status != pyo.SolverStatus.ok:
    print("求解失败！状态码：", results.solver.status)
else:
    # 输出结果
    print("========== 机组启停计划 ==========")
    for g in model.G:
        print(f"\n机组 {g}:")
        for t in model.T:
            u_value = model.u[g, t].value
            if u_value is None:
                print(f"  时段 {t:2d}: 状态=未定义, 出力=未定义")
            else:
                status = "运行" if u_value > 0.5 else "停机"
                p_value = model.p[g, t].value if model.p[g, t].value is not None else "未定义"
                print(f"  时段 {t:2d}: 状态={status}, 出力={p_value} MW")

    print("\n========== 总运行成本 ==========")
    print(f"{model.obj():.2f} 元")