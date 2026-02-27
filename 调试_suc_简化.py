#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SUC模型调试 - 逐步简化约束以找到可行解"""

from 论文第二部分测试 import IEEERTS79System
import pulp
import numpy as np
import time

system = IEEERTS79System()

# 创建简化的SUC1模型 - 逐步增加约束
print("=" * 60)
print("构建简化SUC1模型...")

model = pulp.LpProblem("Simple_SUC", pulp.LpMinimize)

# 只考虑一个场景、一小时、简化设置
T = 3  # 3小时
n_scenarios = 1

# 第一阶段：机组状态（简化：固定开启所有机组）
U = pulp.LpVariable.dicts("U", 
                         ((i, t) for i in range(system.N_GEN) for t in range(T)),
                         cat='Binary')

# 第二阶段：发电量
P = {}
for s in range(n_scenarios):
    P[s] = pulp.LpVariable.dicts(f"P_{s}", 
                                ((i, t) for i in range(system.N_GEN) for t in range(T)),
                                lowBound=0)

# 松弛变量
scenarios = system.get_wind_scenarios(n_scenarios)
max_load = max(sum(system.load_profile[t]) for t in range(T))
max_wind = max(sum(scenarios[0][t]) for t in range(T))

Load_Shed = pulp.LpVariable.dicts("Load_Shed",
                                 (t for t in range(T)),
                                 lowBound=0,
                                 upBound=max_load)

Wind_Curt = pulp.LpVariable.dicts("Wind_Curt",
                                (t for t in range(T)),
                                lowBound=0,
                                upBound=max_wind)

# 目标函数 - 简化版本
cost_expr = pulp.lpSum(P[0][(i, t)] for i in range(system.N_GEN) for t in range(T))

# 加上松弛变量的惩罚
cost_expr += 1000 * pulp.lpSum(Load_Shed[t] for t in range(T))
cost_expr += 50 * pulp.lpSum(Wind_Curt[t] for t in range(T))

model += cost_expr

print(f"\n约束1：机组状态持续性...")
# 约束1：机组状态（简化，对所有机组固定为1）
for i in range(system.N_GEN):
    for t in range(T):
        model += U[(i, t)] == 1, f"State_Fixed_{i}_{t}"

print(f"约束1增加了 {system.N_GEN * T} 个约束")

print(f"\n约束2：功率平衡...")
# 约束2：功率平衡
for t in range(T):
    s = 0
    total_gen = pulp.lpSum(P[s][(i, t)] for i in range(system.N_GEN))
    wind_power = sum(scenarios[s][t])
    total_load = sum(system.load_profile[t])
    
    model += total_gen + wind_power - Wind_Curt[t] - Load_Shed[t] == total_load, f"Power_Balance_{t}"

print(f"约束2增加了 {T} 个约束")

print(f"\n约束3：发电机出力限制...")
# 约束3：发电机出力限制（基于状态）
for i in range(system.N_GEN):
    Pmin = system.generators['Pmin'][i]
    Pmax = system.generators['Pmax'][i]
    
    for t in range(T):
        s = 0
        model += P[s][(i, t)] >= Pmin * U[(i, t)], f"Gen_Min_{i}_{t}"
        model += P[s][(i, t)] <= Pmax * U[(i, t)], f"Gen_Max_{i}_{t}"

print(f"约束3增加了 {2 * system.N_GEN * T} 个约束")

# 求解
print(f"\n总变量数: {len(model.variables())}")
print(f"总约束数: {len(model.constraints)}")

solver = pulp.GUROBI(msg=1, timeLimit=60)
model.solve(solver)

print(f"\n求解结果:")
print(f"状态: {pulp.LpStatus[model.status]}")
if model.status == 1:
    print(f"目标函数值: {pulp.value(model.objective):.2f}")
    
    # 检查功率是否平衡
    for t in range(T):
        total_gen = sum(P[0][(i, t)].varValue or 0 for i in range(system.N_GEN))
        wind_power = sum(scenarios[0][t])
        total_load = sum(system.load_profile[t])
        ls = Load_Shed[t].varValue or 0
        wc = Wind_Curt[t].varValue or 0
        
        balance = total_gen + wind_power - wc - ls
        print(f"Hour {t}: Gen={total_gen:.1f} + Wind={wind_power:.1f} - WC={wc:.1f} - LS={ls:.1f} = Balance={balance:.1f}, Load={total_load:.1f}")
else:
    print(f"无法找到可行解")
