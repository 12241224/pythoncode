#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SUC模型调试 - 移除不合理的初始条件"""

from 论文第二部分测试 import IEEERTS79System
import pulp
import numpy as np

system = IEEERTS79System()

# 创建中等复杂度的SUC1模型
print("=" * 60)
print("构建SUC1模型 - 无不合理的初始条件...")

model = pulp.LpProblem("SUC_Feasible", pulp.LpMinimize)

T = 24
n_scenarios = 1

U = pulp.LpVariable.dicts("U", 
                         ((i, t) for i in range(system.N_GEN) for t in range(T)),
                         cat='Binary')

SU = pulp.LpVariable.dicts("SU", 
                          ((i, t) for i in range(system.N_GEN) for t in range(T)),
                          lowBound=0)

P = {}
for s in range(n_scenarios):
    P[s] = pulp.LpVariable.dicts(f"P_{s}", 
                                ((i, t) for i in range(system.N_GEN) for t in range(T)),
                                lowBound=0)

scenarios = system.get_wind_scenarios(n_scenarios)
max_load = max(sum(system.load_profile[t]) for t in range(T))
max_wind = max(sum(scenarios[0][t]) for t in range(T))

Load_Shed = pulp.LpVariable.dicts("Load_Shed", (t for t in range(T)), 
                                lowBound=0, upBound=max_load)

Wind_Curt = pulp.LpVariable.dicts("Wind_Curt", (t for t in range(T)), 
                                lowBound=0, upBound=max_wind)

# 目标函数
cost_expr = pulp.lpSum(P[0][(i, t)] for i in range(system.N_GEN) for t in range(T))
cost_expr += pulp.lpSum(system.generators['startup_cost'][i] * SU[(i, t)] 
                       for i in range(system.N_GEN) for t in range(T))
cost_expr += 1000 * pulp.lpSum(Load_Shed[t] for t in range(T))
cost_expr += 50 * pulp.lpSum(Wind_Curt[t] for t in range(T))
model += cost_expr

# 约束：启停逻辑  
for i in range(system.N_GEN):
    for t in range(T):
        if t == 0:
            model += SU[(i, t)] >= U[(i, t)], f"Startup_{i}_{t}" 
        else:
            model += SU[(i, t)] >= U[(i, t)] - U[(i, t-1)], f"Startup_{i}_{t}"

# 约束：功率平衡
for t in range(T):
    s = 0
    total_gen = pulp.lpSum(P[s][(i, t)] for i in range(system.N_GEN))
    wind_power = sum(scenarios[s][t])
    total_load = sum(system.load_profile[t])
    
    model += total_gen + wind_power - Wind_Curt[t] - Load_Shed[t] == total_load, f"Power_Balance_{t}"

# 约束：发电机出力限制
for i in range(system.N_GEN):
    Pmin = system.generators['Pmin'][i]
    Pmax = system.generators['Pmax'][i]
    
    for t in range(T):
        s = 0
        model += P[s][(i, t)] >= Pmin * U[(i, t)], f"Gen_Min_{i}_{t}"
        model += P[s][(i, t)] <= Pmax * U[(i, t)], f"Gen_Max_{i}_{t}"

# 不设置初始条件 - 让优化器自由选择

print(f"总变量数: {len(model.variables())}")
print(f"总约束数: {len(model.constraints)}")

solver = pulp.GUROBI(msg=1, timeLimit=120, gapRel=0.05)
model.solve(solver)

print(f"\n求解结果:")
print(f"状态: {pulp.LpStatus[model.status]}")
if model.status == 1:
    print(f"目标函数值: {pulp.value(model.objective):,.0f}")
    
    for t in [0, 6, 12, 18, 23]:
        on_count = sum(1 for i in range(system.N_GEN) if U[(i, t)].varValue and U[(i, t)].varValue > 0.5)
        gen = sum((P[0][(i, t)].varValue or 0) for i in range(system.N_GEN))
        wind = sum(scenarios[0][t])
        load = sum(system.load_profile[t])
        print(f"Hour {t}: {on_count} units on, Gen={gen:.0f}, Wind={wind:.0f}, Load={load:.0f}")
