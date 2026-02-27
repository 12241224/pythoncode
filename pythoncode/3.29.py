import pyomo.environ as pyo
from pyomo.environ import *
import pandas as pd

# ========== 数据预处理 ==========
# 读取所有表格数据
all_sheets = pd.read_excel("C:\\Users\\admin\\Desktop\\3用这个算例-XJTU-ROTS.xlsx", sheet_name=None)

# 提取各表格数据
thermal_units = all_sheets["UnitThermalGenerators"]  # 火电机组
load_profile = all_sheets["LoadCurves"]             # 负荷曲线
transmission_lines = all_sheets["NetLines"]         # 输电线路
wind_units = all_sheets["UnitWindGenerators"]           # 风电机组（假设Sheet名）
solar_units = all_sheets["UnitSolarGenerators"]         # 光伏机组（假设Sheet名）
hydro_units = all_sheets["UnitRunoffHydroGenerators"]         # 水电机组（假设Sheet名）

# 处理负荷计算：总负荷 = sum(负荷有功比例系数 * 负荷曲线值)
# 假设负荷曲线名称对应LoadCurves中的列名
load_curve_data = load_profile.set_index('Hour')  # 假设第一列为Hour

def calculate_total_load():
    total_load = {t: 0 for t in range(24)}
    for _, row in all_sheets["LoadData"].iterrows():  # 假设负荷数据在"LoadData"表中
        curve_name = row["负荷曲线名称"]
        ratio = row["有功比例系数"]
        for t in range(24):
            total_load[t] += ratio * load_curve_data.loc[t, curve_name]
    return total_load

hourly_load = calculate_total_load()

# 处理可再生能源资源曲线（示例，需根据实际数据结构调整）
wind_curve = {
    "风电资源1": [0.8]*24,  # 示例数据，需替换实际曲线
    "风电资源2": [0.7]*24,
    "风电资源3": [0.6]*24
}

solar_curve = {
    "光伏资源1": [0.5]*24,
    "光伏资源2": [0.4]*24,
    "光伏资源3": [0.3]*24
}

hydro_curve = {
    "水电资源1": [0.9]*24,
    "水电资源2": [0.8]*24,
    "水电资源3": [0.7]*24
}

# ========== 建模部分 ==========
model = pyo.ConcreteModel()
model.T = pyo.Set(initialize=range(24))
model.G_thermal = pyo.Set(initialize=thermal_units.index)  # 火电机组集合
model.G_wind = pyo.Set(initialize=wind_units.index)        # 风电机组集合
model.G_solar = pyo.Set(initialize=solar_units.index)      # 光伏机组集合
model.G_hydro = pyo.Set(initialize=hydro_units.index)      # 水电机组集合
model.Lines = pyo.Set(initialize=transmission_lines.index)

# 定义变量
model.u = pyo.Var(model.G_thermal, model.T, within=pyo.Binary)  # 火电启停
model.p_thermal = pyo.Var(model.G_thermal, model.T, within=pyo.NonNegativeReals)  # 火电出力
model.p_wind = pyo.Var(model.G_wind, model.T, within=pyo.NonNegativeReals)        # 风电出力
model.p_solar = pyo.Var(model.G_solar, model.T, within=pyo.NonNegativeReals)      # 光伏出力
model.p_hydro = pyo.Var(model.G_hydro, model.T, within=pyo.NonNegativeReals)      # 水电出力

# 目标函数（火电成本）
def objective_rule(model):
    cost = 0
    for g in model.G_thermal:
        startup_cost = thermal_units.loc[g, "启动费用(吨标准煤)"]
        A = thermal_units.loc[g, "运行成本曲线系数A(吨标准煤/(MW)^2)"] * 1e-4 * 1500
        B = thermal_units.loc[g, "运行成本曲线系数B(吨标准煤/MW)"] * 1e-2 * 1500
        C_val = thermal_units.loc[g, "运行成本曲线系数C(吨标准煤)"] * 1500
        for t in model.T:
            cost += startup_cost * model.u[g, t]  # 假设简化启动逻辑
            cost += (A * model.p_thermal[g, t]**2 + B * model.p_thermal[g, t] + C_val)
    return cost

model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# 可再生能源出力约束
def wind_power_rule(model, g, t):
    curve_name = wind_units.loc[g, "风电资源曲线名称"]
    return model.p_wind[g, t] <= wind_units.loc[g, "机组装机容量(MW)"] * wind_curve[curve_name][t]

model.wind_power = pyo.Constraint(model.G_wind, model.T, rule=wind_power_rule)

def solar_power_rule(model, g, t):
    curve_name = solar_units.loc[g, "光伏资源曲线名称"]
    return model.p_solar[g, t] <= solar_units.loc[g, "机组装机容量(MW)"] * solar_curve[curve_name][t]

model.solar_power = pyo.Constraint(model.G_solar, model.T, rule=solar_power_rule)

def hydro_power_rule(model, g, t):
    curve_name = hydro_units.loc[g, "径流式水电资源曲线名称"]
    return model.p_hydro[g, t] <= hydro_units.loc[g, "机组装机容量(MW)"] * hydro_curve[curve_name][t]

model.hydro_power = pyo.Constraint(model.G_hydro, model.T, rule=hydro_power_rule)

# 功率平衡约束（包含所有机组类型）
def power_balance_rule(model, t):
    total_gen = (
        sum(model.p_thermal[g, t] for g in model.G_thermal) +
        sum(model.p_wind[g, t] for g in model.G_wind) +
        sum(model.p_solar[g, t] for g in model.G_solar) +
        sum(model.p_hydro[g, t] for g in model.G_hydro)
    )
    return total_gen == hourly_load[t]

model.power_balance = pyo.Constraint(model.T, rule=power_balance_rule)

# 火电机组其他约束
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

# ======== 求解与结果输出 ========
solver = pyo.SolverFactory('cbc')
results = solver.solve(model)

# 结果输出（示例）
if results.solver.status == pyo.SolverStatus.ok:
    print("总成本:", model.obj())
    for t in model.T:
        print(f"时段 {t}: 总负荷={hourly_load[t]:.2f} MW")
        print("火电出力:", sum(model.p_thermal[g, t].value for g in model.G_thermal))
        print("风电出力:", sum(model.p_wind[g, t].value for g in model.G_wind))
        print("光伏出力:", sum(model.p_solar[g, t].value for g in model.G_solar))
        print("水电出力:", sum(model.p_hydro[g, t].value for g in model.G_hydro))
else:
    print("求解失败")