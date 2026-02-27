#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最小化SUC模型 - 仅包含基本约束"""

from 论文第二部分测试 import IEEERTS79System
import pulp

system = IEEERTS79System()

print("构建最小化SUC模型...")

model = pulp.LpProblem("Minimal_SUC", pulp.LpMinimize)

T = 24
n_GEN = system.N_GEN
n_scenarios = 1

# 第一阶段：机组状态
U = pulp.LpVariable.dicts("U", 
                         ((i, t) for i in range(n_GEN) for t in range(T)),
                         cat='Binary')

# 第二阶段：发电量  
P = {}
for s in range(n_scenarios):
    P[s] = pulp.LpVariable.dicts(f"P_{s}", 
                                ((i, t) for i in range(n_GEN) for t in range(T)),
                                lowBound=0)

scenarios = system.get_wind_scenarios(n_scenarios)

# 松弛变量（宽松限制）
Load_Shed = pulp.LpVariable.dicts("Load_Shed", (t for t in range(T)), lowBound=0)
Wind_Curt = pulp.LpVariable.dicts("Wind_Curt", (t for t in range(T)), lowBound=0)

# 目标函数 - 最小化总成本
cost_expr = pulp.lpSum(P[0][(i, t)] for i in range(n_GEN) for t in range(T))
cost_expr += 1000 * pulp.lpSum(Load_Shed[t] for t in range(T))
cost_expr += 50 * pulp.lpSum(Wind_Curt[t] for t in range(T))
model += cost_expr

print("添加约束...")

# 约束1：功率平衡（仅约束）
for t in range(T):
    s = 0
    total_gen = pulp.lpSum(P[s][(i, t)] for i in range(n_GEN))
    wind_power = sum(scenarios[s][t])
    total_load = sum(system.load_profile[t])
    
    model += total_gen + wind_power - Wind_Curt[t] - Load_Shed[t] == total_load, f"Power_Balance_{t}"

# 约束2：发电机出力限制
for i in range(n_GEN):
    Pmin = system.generators['Pmin'][i]
    Pmax = system.generators['Pmax'][i]
    
    for t in range(T):
        s = 0
        # 关键：这里必须用 U[(i, t)] 作为启停变量
        model += P[s][(i, t)] >= Pmin * U[(i, t)], f"Gen_Min_{i}_{t}"
        model += P[s][(i, t)] <= Pmax * U[(i, t)], f"Gen_Max_{i}_{t}"

print(f"变量数: {len(model.variables())}")
print(f"约束数: {len(model.constraints)}")

print("\n求解...")
solver = pulp.GUROBI(msg=1, timeLimit=120)
model.solve(solver)

print(f"\n结果: {pulp.LpStatus[model.status]}")

if model.status == 1:
    print(f"目标函数值: {pulp.value(model.objective):,.0f}")
    
    for t in [0, 6, 12, 18, 23]:
        on_count = sum(1 for i in range(n_GEN) if U[(i, t)].varValue and U[(i, t)].varValue > 0.5)
        gen = sum((P[0][(i, t)].varValue or 0) for i in range(n_GEN))
        wind = sum(scenarios[0][t])
        load = sum(system.load_profile[t])
        print(f"Hour {t}: {on_count} units on, Gen={gen:.0f}, Wind={wind:.0f}, Load={load:.0f}")
