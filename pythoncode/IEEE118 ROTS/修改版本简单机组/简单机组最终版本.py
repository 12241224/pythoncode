import pyomo.environ as pyo
from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import pandas as pd
import os
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
import itertools

# ========== 输入部分（从Excel读取数据） ==========
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建完整的文件路径
excel_path = os.path.join(script_dir, "IEEE30.xlsm")

# 检查文件是否存在
if not os.path.exists(excel_path):
    raise FileNotFoundError(f"文件 {excel_path} 不存在，请检查文件路径是否正确！")

# ===== 1. 读取网络数据 =====
print("正在读取网络数据...")
try:
    # 1.1 读取母线数据
    buses_df = pd.read_excel(excel_path, sheet_name="NetBuses")
    bus_names = buses_df['母线编号'].tolist()
    bus_name_to_id = {name: idx for idx, name in enumerate(bus_names)}
    num_buses = len(bus_names)
    
    # 1.2 读取支路数据
    branches_df = pd.read_excel(excel_path, sheet_name="NetBranches")
    branches = []
    for _, row in branches_df.iterrows():
        from_bus = bus_name_to_id[row['首端母线编号']]
        to_bus = bus_name_to_id[row['末端母线编号']]
        reactance = row['线路电抗(pu)']
        capacity = row['传输容量极限(100MW)'] * 100  # 转换为MW
        branches.append({
            'from': from_bus,
            'to': to_bus,
            'reactance': reactance,
            'capacity': capacity
        })
    
    # 1.3 构建节点导纳矩阵B
    B_matrix = np.zeros((num_buses, num_buses))
    for branch in branches:
        i, j = branch['from'], branch['to']
        b_ij = 1 / branch['reactance']
        B_matrix[i, i] -= b_ij
        B_matrix[j, j] -= b_ij
        B_matrix[i, j] = b_ij
        B_matrix[j, i] = b_ij
    
    # 1.4 计算转移分布因子矩阵F
    # 选择第一个节点作为参考节点（相角为0）
    B_reduced = B_matrix[1:, 1:]
    X_reduced = np.linalg.inv(B_reduced)
    
    # 扩展X矩阵，参考节点对应的行列为0
    X_full = np.zeros((num_buses, num_buses))
    X_full[1:, 1:] = X_reduced
    
    # 计算转移分布因子F
    F_matrix = {}
    for idx, branch in enumerate(branches):
        i, j = branch['from'], branch['to']
        F_row = {}
        for bus_idx in range(num_buses):
            F_row[bus_idx] = (X_full[i, bus_idx] - X_full[j, bus_idx]) / branch['reactance']
        F_matrix[idx] = F_row
    
    print("转移分布因子矩阵计算完成")

except Exception as e:
    print(f"读取网络数据时出错: {e}")
    raise

# ===== 2. 读取负荷数据 =====
print("正在读取负荷数据...")
try:
    # 2.1 读取负荷分布数据
    loads_df = pd.read_excel(excel_path, sheet_name="Loads")
    
    # 创建负荷比例系数字典（母线ID:比例系数）
    load_scale = {}
    for _, row in loads_df.iterrows():
        bus_name = row['母线编号']
        bus_id = bus_name_to_id[bus_name]
        scale = float(row['有功比例系数（乘以对应资源曲线即可得真实负荷值）'])
        load_scale[bus_id] = scale
    
    # 2.2 读取负荷曲线数据
    load_curve_df = pd.read_excel(excel_path, sheet_name="LoadCurves")
    
    # 创建负荷因子字典
    load_factor = {}
    for _, row in load_curve_df.iterrows():
        hour = int(row['Hour'])
        factor = float(row['LoadFactor'])
        load_factor[hour] = factor
    
    # 2.3 计算每个时刻每个母线的负荷
    base_load = 100  # 基准负荷为100MW
    bus_hourly_load = {bus_id: {} for bus_id in range(num_buses)}
    hourly_total_load = {}
    
    for t in range(24):
        total_load = 0
        for bus_id, scale in load_scale.items():
            load_value = scale * load_factor[t] * base_load
            bus_hourly_load[bus_id][t] = load_value
            total_load += load_value
        hourly_total_load[t] = total_load
    
    print(f"系统峰值负荷: {max(hourly_total_load.values()):.2f} MW")
    print(f"系统谷值负荷: {min(hourly_total_load.values()):.2f} MW")

except Exception as e:
    print(f"读取负荷数据时出错: {e}")
    raise

# ===== 3. 读取机组参数 =====
print("正在读取机组参数...")
try:
    gen_df = pd.read_excel(excel_path, sheet_name="UnitThermalGenerators")
    
    # 创建机组参数字典
    generators = {}
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 预设颜色
    
    for idx, row in gen_df.iterrows():
        gen_name = row['火电机组名称']
        bus_name = row['所在母线编号']
        bus_id = bus_name_to_id[bus_name]
        
        # 创建机组参数字典
        gen_params = {
            'bus': bus_id,
            'Pmax': row['装机容量(100MW)'] * 100,  # 转换为MW
            'Pmin': row['最小技术出力(100MW)'] * 100,  # 转换为MW
            'min_up_time': row['最短开机时间(h)'],
            'min_down_time': row['最短关机时间(h)'],
            'ramp_up': row['上爬坡速率(100MW/h)'] * 100,  # 转换为MW/h
            'ramp_down': row['下爬坡速率(100MW/h)'] * 100,  # 转换为MW/h
            'startup_cost': row['启动费用(元)'],
            'shutdown_cost': row['停机费用（元）'],
            'a': row['运行成本曲线系数A(元/(100MW)^2)'],
            'b': row['运行成本曲线系数B(元/100MW)'],
            'c': row['运行成本曲线系数C(元)'],
            'color': colors[idx % len(colors)]  # 使用预设颜色
        }
        
        generators[gen_name] = gen_params

    print(f"成功读取 {len(generators)} 台机组参数")

except Exception as e:
    print(f"读取机组参数时出错: {e}")
    raise

# ========== 建模部分 ==========
print("开始构建优化模型...")
model = pyo.ConcreteModel()

# 定义集合
model.G = pyo.Set(initialize=generators.keys())  # 机组集合
model.T = pyo.Set(initialize=range(24))          # 时间集合
model.B = pyo.Set(initialize=range(num_buses))   # 母线集合
model.L = pyo.Set(initialize=range(len(branches))) # 支路集合

# 定义变量
model.u = pyo.Var(model.G, model.T, within=pyo.Binary)     # 机组启停状态
model.p = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)  # 机组出力（MW）
model.su = pyo.Var(model.G, model.T, within=pyo.Binary)    # 启动标志
model.sd = pyo.Var(model.G, model.T, within=pyo.Binary)    # 停机标志
model.theta = pyo.Var(model.B, model.T, within=pyo.Reals)  # 母线电压相角

# 辅助变量（单位转换为100MW，用于成本计算）
model.p_100mw = pyo.Var(model.G, model.T, within=pyo.NonNegativeReals)  # 机组出力（100MW）

# 目标函数：总成本 = 运行成本 + 启停成本
def objective_rule(model):
    total_cost = 0
    for g in model.G:
        for t in model.T:
            # 运行成本（注意：常数项乘以状态变量）
            total_cost += (generators[g]['a'] * model.p_100mw[g, t]**2 + 
                          generators[g]['b'] * model.p_100mw[g, t] + 
                          generators[g]['c'] * model.u[g, t])  # 修正点：c * u[g,t]
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

# 3. 修正的启停逻辑约束
def startup_logic_rule(model, g, t):
    if t == 0:
        # 初始状态假设为停机
        return model.u[g, t] == model.su[g, t]
    return model.u[g, t] - model.u[g, t-1] == model.su[g, t] - model.sd[g, t]

model.startup_logic = pyo.Constraint(model.G, model.T, rule=startup_logic_rule)

# 4. 修正的爬坡约束
def ramp_up_rule(model, g, t):
    if t == 0:
        return pyo.Constraint.Skip
    return (model.p[g, t] - model.p[g, t-1] <= 
            generators[g]['ramp_up'] + model.su[g, t] * generators[g]['Pmax'])

def ramp_down_rule(model, g, t):
    if t == 0:
        return pyo.Constraint.Skip
    return (model.p[g, t-1] - model.p[g, t] <= 
            generators[g]['ramp_down'] + model.sd[g, t] * generators[g]['Pmax'])

model.ramp_up = pyo.Constraint(model.G, model.T, rule=ramp_up_rule)
model.ramp_down = pyo.Constraint(model.G, model.T, rule=ramp_down_rule)

# 5. 最小启停时间约束
def min_up_time_rule(model, g, t):
    min_up = generators[g]['min_up_time']
    if t < min_up:
        return pyo.Constraint.Skip
    return sum(model.u[g, k] for k in range(t, min(t + min_up, 24))) >= min_up * model.su[g, t]

def min_down_time_rule(model, g, t):
    min_down = generators[g]['min_down_time']
    if t < min_down:
        return pyo.Constraint.Skip
    return sum(1 - model.u[g, k] for k in range(t, min(t + min_down, 24))) >= min_down * model.sd[g, t]

model.min_up_time = pyo.Constraint(model.G, model.T, rule=min_up_time_rule)
model.min_down_time = pyo.Constraint(model.G, model.T, rule=min_down_time_rule)

# 6. 节点功率平衡约束
def node_power_balance_rule(model, b, t):
    gen_power = sum(model.p[g, t] for g in model.G if generators[g]['bus'] == b)
    load_power = -bus_hourly_load[b].get(t, 0.0)
    
    line_flow = 0.0
    for idx, branch in enumerate(branches):
        if branch['from'] == b:
            line_flow += (model.theta[b, t] - model.theta[branch['to'], t]) / branch['reactance']
        elif branch['to'] == b:
            line_flow += (model.theta[b, t] - model.theta[branch['from'], t]) / branch['reactance']
    
    return gen_power + load_power + line_flow == 0

model.node_power_balance = pyo.Constraint(model.B, model.T, rule=node_power_balance_rule)

# 7. 参考节点相角设为0
def reference_bus_rule(model, t):
    return model.theta[0, t] == 0

model.reference_bus = pyo.Constraint(model.T, rule=reference_bus_rule)

# 8. 线路传输容量约束
def line_capacity_rule(model, l, t):
    branch = branches[l]
    flow = 0.0
    for b in model.B:
        net_injection = 0.0
        for g in model.G:
            if generators[g]['bus'] == b:
                net_injection += model.p[g, t]
        net_injection -= bus_hourly_load[b].get(t, 0.0)
        flow += F_matrix[l].get(b, 0.0) * net_injection
    return (-branch['capacity'], flow, branch['capacity'])

model.line_capacity = pyo.Constraint(model.L, model.T, rule=line_capacity_rule)

# 9. 添加备用约束（5%备用）
def reserve_constraint_rule(model, t):
    total_reserve = 0.0
    for g in model.G:
        total_reserve += model.u[g, t] * generators[g]['Pmax'] - model.p[g, t]
    return total_reserve >= 0.05 * hourly_total_load[t]

model.reserve_constraint = pyo.Constraint(model.T, rule=reserve_constraint_rule)

# ========== 保存模型为LP文件 ==========
print("正在将模型保存为LP文件...")
model.write('uc_model.lp', io_options={'symbolic_solver_labels': True})
print("模型已保存为 uc_model.lp")

# ========== 使用Gurobi求解 ==========
solver = pyo.SolverFactory('gurobi')
solver.options['MIPGap'] = 0.001
solver.options['TimeLimit'] = 600
solver.options['LogFile'] = 'gurobi.log'
solver.options['Presolve'] = 2
solver.options['Heuristics'] = 0.05
solver.options['Cuts'] = 2

print("开始求解模型...")
results = solver.solve(model, tee=True)

# 检查求解状态
if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
    print("求解成功！")
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print("模型不可行！")
    solver.options['IISFile'] = 'uc_model.ilp'
    solver.solve(model, tee=True, load_solutions=False)
    print("IIS已保存为 uc_model.ilp")
    exit()
else:
    print("求解未完成！状态:", results.solver.status)
    exit()

# ========== 结果处理与输出 ==========
# 修正的成本计算函数
def calculate_generation_cost(g, t):
    if pyo.value(model.u[g, t]) > 0.5:  # 机组运行
        p_100mw_val = pyo.value(model.p_100mw[g, t])
        return (generators[g]['a'] * p_100mw_val**2 + 
                generators[g]['b'] * p_100mw_val + 
                generators[g]['c'])
    else:  # 机组停机
        return 0.0

# 打印结果
print("\n========== 机组启停计划 ==========")
for g in model.G:
    print(f"\n机组 {g}:")
    for t in model.T:
        status = "运行" if pyo.value(model.u[g, t]) > 0.5 else "停机"
        print(f"  时段 {t:2d}: 状态={status}, 出力={pyo.value(model.p[g, t]):.2f} MW")

total_cost = pyo.value(model.obj)
print("\n========== 总运行成本 ==========")
print(f"{total_cost:.2f} 元")

# ========== 结果可视化 ==========
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建时间轴（假设从2025-07-18 00:00开始）
start_date = datetime(2025, 7, 18, 0, 0)
hours = [start_date + timedelta(hours=t) for t in model.T]

# 1. 负荷曲线与机组出力堆叠图
fig1, ax1 = plt.subplots(figsize=(14, 8))

# 准备堆叠数据
stack_data = {g: [pyo.value(model.p[g, t]) for t in model.T] for g in model.G}
bottom = np.zeros(24)

# 绘制每个机组的出力
for g in model.G:
    ax1.bar(hours, stack_data[g], bottom=bottom, width=0.03, 
            label=f'{g} (最大:{generators[g]["Pmax"]}MW)', 
            color=generators[g]['color'])
    bottom += np.array(stack_data[g])

# 绘制总负荷曲线
ax1.plot(hours, [hourly_total_load[t] for t in model.T], 
         'ko-', linewidth=2, markersize=6, label='总负荷')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax1.set_xlabel('时间')
ax1.set_ylabel('出力 (MW)')
ax1.set_title('机组出力与系统负荷')
ax1.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
ax1.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

# 2. 机组启停状态图
fig2, axs = plt.subplots(len(model.G), 1, figsize=(14, 12), sharex=True)
if len(model.G) == 1:
    axs = [axs]  # 如果只有一台机组，确保axs是列表

for i, g in enumerate(model.G):
    status = [pyo.value(model.u[g, t]) for t in model.T]
    axs[i].step(hours, status, where='post', 
               linewidth=2, color=generators[g]['color'])
    axs[i].set_ylim(-0.1, 1.1)
    axs[i].set_yticks([0, 1])
    axs[i].set_yticklabels(['停机', '运行'])
    axs[i].set_ylabel(g)
    axs[i].grid(True, linestyle='--', alpha=0.7)

axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axs[-1].set_xlabel('时间')
plt.suptitle('机组启停状态')
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# 3. 机组成本构成分析（修正版）
generation_costs = {}
for g in model.G:
    # 只计算机组运行时段的运行成本
    gen_cost = 0
    for t in model.T:
        if pyo.value(model.u[g, t]) > 0.5:  # 机组运行时
            p_100mw_value = pyo.value(model.p_100mw[g, t])
            gen_cost += (generators[g]['a'] * p_100mw_value**2 + 
                        generators[g]['b'] * p_100mw_value + 
                        generators[g]['c'])
    
    startup_cost = sum(generators[g]['startup_cost'] * pyo.value(model.su[g, t]) for t in model.T)
    shutdown_cost = sum(generators[g]['shutdown_cost'] * pyo.value(model.sd[g, t]) for t in model.T)
    generation_costs[g] = {
        '运行成本': gen_cost,
        '启动成本': startup_cost,
        '停机成本': shutdown_cost,
        '总成本': gen_cost + startup_cost + shutdown_cost
    }

# 创建成本构成数据
categories = ['运行成本', '启动成本', '停机成本']
cost_data = {g: [generation_costs[g][cat] for cat in categories] for g in model.G}

# 过滤掉总成本为 0 的机组（用于饼图）
active_generators = {g: generation_costs[g]['总成本'] for g in model.G if generation_costs[g]['总成本'] > 0}
total_costs = list(active_generators.values())
active_gen_names = list(active_generators.keys())
colors = [generators[g]['color'] for g in active_gen_names]

fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# 成本构成堆叠图（显示所有机组，包括成本为0的）
bottom = np.zeros(len(model.G))
for i, cat in enumerate(categories):
    values = [generation_costs[g][cat] for g in model.G]
    ax3.bar(model.G, values, bottom=bottom, label=cat)
    bottom += values

ax3.set_title('机组成本构成')
ax3.set_ylabel('成本 (元)')
ax3.legend()

# 总成本饼图（只显示有成本的机组）
if len(active_gen_names) > 0:
    ax4.pie(total_costs, labels=active_gen_names, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax4.set_title('总成本分布')
    ax4.axis('equal')
else:
    ax4.text(0.5, 0.5, '所有机组成本为0', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('总成本分布')

plt.tight_layout()

# 4. 系统运行摘要
fig4, ax5 = plt.subplots(figsize=(12, 6))
ax5.axis('off')  # 关闭坐标轴

# 创建摘要文本
summary_text = (
    f"系统总运行成本: {total_cost:.2f} 元\n\n"
    f"机组运行小时数:\n"
)

# 存储汇总数据用于Excel
summary_data = []

for g in model.G:
    running_hours = sum(1 for t in model.T if pyo.value(model.u[g, t]) > 0.5)
    startup_count = sum(pyo.value(model.su[g, t]) for t in model.T)
    shutdown_count = sum(pyo.value(model.sd[g, t]) for t in model.T)
    avg_output = np.mean([pyo.value(model.p[g, t]) for t in model.T if pyo.value(model.u[g, t]) > 0.5])
    avg_output = avg_output if not np.isnan(avg_output) else 0
    
    summary_text += (
        f"  {g}: 运行小时 = {running_hours}/24, 启动次数 = {startup_count:.0f}, "
        f"停机次数 = {shutdown_count:.0f}, 平均出力 = {avg_output:.2f} MW\n"
    )

    # 添加到汇总数据
    summary_data.append({
        '机组': g,
        '运行小时数': running_hours,
        '启动次数': startup_count,
        '停机次数': shutdown_count,
        '平均出力(MW)': avg_output,
        '总成本(元)': generation_costs[g]['总成本'],
        '运行成本(元)': generation_costs[g]['运行成本'],
        '启动成本(元)': generation_costs[g]['启动成本'],
        '停机成本(元)': generation_costs[g]['停机成本']
    })

summary_text += "\n成本明细:\n"
for g in model.G:
    summary_text += (
        f"  {g}: 总成本 = {generation_costs[g]['总成本']:.2f} 元 "
        f"(运行: {generation_costs[g]['运行成本']:.2f}, "
        f"启动: {generation_costs[g]['启动成本']:.2f}, "
        f"停机: {generation_costs[g]['停机成本']:.2f})\n"
    )

ax5.text(0.05, 0.95, summary_text, fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax5.set_title('系统运行摘要')

plt.tight_layout()

# 显示所有图表
plt.show()

# ========== 将结果保存到Excel ==========
print("\n正在将结果保存到Excel文件...")

# 创建Excel写入对象
with pd.ExcelWriter('机组组合优化结果.xlsx') as writer:
    # 1. 机组调度计划表
    schedule_data = []
    for t in model.T:
        row = {'时段': t, '系统负荷(MW)': hourly_total_load[t]}
        for g in model.G:
            row[f'{g}_状态'] = '运行' if pyo.value(model.u[g, t]) > 0.5 else '停机'
            row[f'{g}_出力(MW)'] = pyo.value(model.p[g, t])
            row[f'{g}_启动标志'] = pyo.value(model.su[g, t])
            row[f'{g}_停机标志'] = pyo.value(model.sd[g, t])
        schedule_data.append(row)
    
    schedule_df = pd.DataFrame(schedule_data)
    schedule_df.to_excel(writer, sheet_name='机组调度计划', index=False)
    
    # 2. 机组成本明细表
    cost_details = []
    for g in model.G:
        for t in model.T:
            # 使用修正的成本计算函数
            var_cost = calculate_generation_cost(g, t)
            start_cost = generators[g]['startup_cost'] * pyo.value(model.su[g, t])
            shutdown_cost = generators[g]['shutdown_cost'] * pyo.value(model.sd[g, t])
            total_cost = var_cost + start_cost + shutdown_cost
            
            cost_details.append({
                '机组': g,
                '时段': t,
                '出力(MW)': pyo.value(model.p[g, t]),
                '状态': '运行' if pyo.value(model.u[g, t]) > 0.5 else '停机',
                '运行成本(元)': var_cost,
                '启动成本(元)': start_cost,
                '停机成本(元)': shutdown_cost,
                '总成本(元)': total_cost
            })
    
    cost_df = pd.DataFrame(cost_details)
    cost_df.to_excel(writer, sheet_name='机组成本明细', index=False)
    
    # 3. 机组汇总信息表
    summary_data = []
    for g in model.G:
        running_hours = sum(1 for t in model.T if pyo.value(model.u[g, t]) > 0.5)
        startup_count = sum(pyo.value(model.su[g, t]) for t in model.T)
        shutdown_count = sum(pyo.value(model.sd[g, t]) for t in model.T)
        avg_output = np.mean([pyo.value(model.p[g, t]) for t in model.T if pyo.value(model.u[g, t]) > 0.5])
        avg_output = avg_output if not np.isnan(avg_output) else 0
        
        # 计算总运行成本（使用修正的成本计算函数）
        gen_cost = sum(calculate_generation_cost(g, t) for t in model.T)
        startup_cost = sum(generators[g]['startup_cost'] * pyo.value(model.su[g, t]) for t in model.T)
        shutdown_cost = sum(generators[g]['shutdown_cost'] * pyo.value(model.sd[g, t]) for t in model.T)
        total_gen_cost = gen_cost + startup_cost + shutdown_cost
        
        summary_data.append({
            '机组': g,
            '运行小时数': running_hours,
            '启动次数': startup_count,
            '停机次数': shutdown_count,
            '平均出力(MW)': avg_output,
            '总成本(元)': total_gen_cost,
            '运行成本(元)': gen_cost,
            '启动成本(元)': startup_cost,
            '停机成本(元)': shutdown_cost
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name='机组汇总信息', index=False)
    
    # 4. 系统总览表
    overview_data = [{
        '总运行成本(元)': total_cost,
        '总启动次数': sum(pyo.value(model.su[g, t]) for g in model.G for t in model.T),
        '总停机次数': sum(pyo.value(model.sd[g, t]) for g in model.G for t in model.T),
        '系统最大负荷(MW)': max(hourly_total_load.values()),
        '系统最小负荷(MW)': min(hourly_total_load.values()),
        '系统平均负荷(MW)': np.mean(list(hourly_total_load.values()))
    }]
    
    overview_df = pd.DataFrame(overview_data)
    overview_df.to_excel(writer, sheet_name='系统总览', index=False)

print("结果已保存到 '机组组合优化结果.xlsx'")