#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""最小化SUC模型 - 3个场景版本"""

from 论文第二部分测试 import IEEERTS79System
import pulp

system = IEEERTS79System()

print("构建最小化SUC模型 - 3场景...")

model = pulp.LpProblem("Minimal_SUC_3Scenario", pulp.LpMinimize)

T = 24
n_GEN = system.N_GEN
n_scenarios = 3

# 第一阶段：机组状态（所有场景相同）
U = pulp.LpVariable.dicts("U", 
                         ((i, t) for i in range(n_GEN) for t in range(T)),
                         cat='Binary')

# 第二阶段：发电量（每个场景不同）
P = {}
for s in range(n_scenarios):
    P[s] = pulp.LpVariable.dicts(f"P_{s}", 
                                ((i, t) for i in range(n_GEN) for t in range(T)),
                                lowBound=0)

scenarios = system.get_wind_scenarios(n_scenarios)

# 松弛变量
Load_Shed = {}
Wind_Curt = {}
for s in range(n_scenarios):
    Load_Shed[s] = pulp.LpVariable.dicts(f"Load_Shed_{s}", (t for t in range(T)), lowBound=0)
    Wind_Curt[s] = pulp.LpVariable.dicts(f"Wind_Curt_{s}", (t for t in range(T)), lowBound=0)

# 目标函数
cost_expr = 0
for s in range(n_scenarios):
    scenario_prob = 1.0 / n_scenarios
    cost_expr += scenario_prob * pulp.lpSum(P[s][(i, t)] for i in range(n_GEN) for t in range(T))
    cost_expr += scenario_prob * 1000 * pulp.lpSum(Load_Shed[s][t] for t in range(T))
    cost_expr += scenario_prob * 50 * pulp.lpSum(Wind_Curt[s][t] for t in range(T))

# 第一阶段成本（与s无关）
cost_expr += pulp.lpSum(10 * (U[(i, t)] - (U[(i, t-1)] if t > 0 else 0)) for i in range(n_GEN) for t in range(T))

model += cost_expr

print("添加约束...")

# 对每个场景添加约束
for s in range(n_scenarios):
    # 功率平衡
    for t in range(T):
        total_gen = pulp.lpSum(P[s][(i, t)] for i in range(n_GEN))
        wind_power = sum(scenarios[s][t])
        total_load = sum(system.load_profile[t])
        
        model += total_gen + wind_power - Wind_Curt[s][t] - Load_Shed[s][t] == total_load, f"Power_Balance_{s}_{t}"
    
    # 发电机出力限制
    for i in range(n_GEN):
        Pmin = system.generators['Pmin'][i]
        Pmax = system.generators['Pmax'][i]
        
        for t in range(T):
            model += P[s][(i, t)] >= Pmin * U[(i, t)], f"Gen_Min_{s}_{i}_{t}"
            model += P[s][(i, t)] <= Pmax * U[(i, t)], f"Gen_Max_{s}_{i}_{t}"

print(f"变量数: {len(model.variables())}")
print(f"约束数: {len(model.constraints)}")

print("\n求解...")
solver = pulp.GUROBI(msg=1, timeLimit=120, gapRel=0.05)
model.solve(solver)

print(f"\n结果: {pulp.LpStatus[model.status]}")

if model.status == 1:
    print(f"目标函数值: {pulp.value(model.objective):,.0f}")
