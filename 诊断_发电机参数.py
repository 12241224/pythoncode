#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发电机参数诊断
检查发电机的cost_a、cost_b、startup_cost等参数是否合理
"""

import sys
sys.path.insert(0, r'd:\pythoncode')

from 论文第二部分测试 import IEEERTS79System
import pandas as pd

print("=" * 70)
print("发电机成本参数诊断")
print("=" * 70)

system = IEEERTS79System()

print(f"\n【IEEE RTS-79系统发电机参数】")
print(f"共{system.N_GEN}台机组\n")

# 提取关键参数
data = {
    '机组ID': [],
    '机组类型': [],
    'Pmin(MW)': [],
    'Pmax(MW)': [],
    'cost_b($/MW)': [],
    'cost_a($/MW²)': [],
    '启动成本($)': [],
    '固定成本($/h)': [],
}

for i in range(system.N_GEN):
    gen_type = system.generators['type'].iloc[i]
    Pmin = system.generators['Pmin'][i]
    Pmax = system.generators['Pmax'][i]
    cost_b = system.generators['cost_b'][i]
    cost_a = system.generators.get('cost_a', [0]*system.N_GEN)[i]
    startup = system.generators['startup_cost'][i]
    fixed = system.generators.get('cost_c', [0]*system.N_GEN)[i]
    
    data['机组ID'].append(i)
    data['机组类型'].append(gen_type)
    data['Pmin(MW)'].append(f"{Pmin:.1f}")
    data['Pmax(MW)'].append(f"{Pmax:.1f}")
    data['cost_b($/MW)'].append(f"{cost_b:.1f}")
    data['cost_a($/MW²)'].append(f"{cost_a:.4f}")
    data['启动成本($)'].append(f"{startup:.0f}")
    data['固定成本($/h)'].append(f"{fixed:.1f}")

df = pd.DataFrame(data)
print(df.to_string(index=False))

print(f"\n【成本分析】")
print("-" * 70)

# 计算平均成本
cost_b_list = system.generators['cost_b'].tolist()
cost_a_list = system.generators.get('cost_a', [0]*system.N_GEN)
startup_list = system.generators['startup_cost'].tolist()
fixed_list = system.generators.get('cost_c', [0]*system.N_GEN)

print(f"cost_b (边际成本)统计:")
print(f"  最小: {min(cost_b_list):.2f} $/MW")
print(f"  最大: {max(cost_b_list):.2f} $/MW")
print(f"  平均: {sum(cost_b_list)/len(cost_b_list):.2f} $/MW")

print(f"\nstartup_cost (启动成本)统计:")
print(f"  最小: {min(startup_list):.0f} $")
print(f"  最大: {max(startup_list):.0f} $")
print(f"  平均: {sum(startup_list)/len(startup_list):.0f} $")

print(f"\n【关键对比】")
print("-" * 70)
print(f"失负荷惩罚成本 (theta_ens): 50 $/MWh")
print(f"边际成本范围: {min(cost_b_list):.1f} - {max(cost_b_list):.1f} $/MWh")
print(f"平均边际成本: {sum(cost_b_list)/len(cost_b_list):.2f} $/MWh")

print(f"\n【诊断】")
print("-" * 70)

avg_cost_b = sum(cost_b_list) / len(cost_b_list)
if avg_cost_b > 50:
    print(f"✓ 边际成本({avg_cost_b:.1f})远高于失负荷惩罚(50)")
    print(f"  -> 优化器倾向于增加备用而不是允许失负荷")
    print(f"  -> 这可能导致EENS=0（行为正确的经济决策）")
else:
    print(f"⚠ 边际成本({avg_cost_b:.1f})低于或等于失负荷惩罚(50)")
    print(f"  -> 允许少量失负荷会更经济")
    print(f"  -> EENS应该>0，但目前为0可能是约束设定问题")

# 完整性计算：1MW备用成本
print(f"\n【备用成本分析】")
print("-" * 70)
reserve_cost_per_mw = 5  # $/MW
print(f"1MW备用成本 (系统参数): {reserve_cost_per_mw} $/MWh")
print(f"1MWh失负荷成本: {50} $/MWh")
print(f"备用成本比例: {reserve_cost_per_mw/50*100:.1f}%")

# 对于Pmax=200的机组，运维一整天
print(f"\n【示例计算：容量200MW机组24小时运维】")
print("-" * 70)
ex_pmax = 200
ex_cost_b = 15  # 示例
ex_cost_a = 0.006  # 示例
ex_fixed = 500  # 示例

# 平均出力
avg_output = ex_pmax * 0.6  # 假设平均负荷率60%
quadratic_cost_per_mw = ex_cost_a * avg_output
linear_cost_per_mw = ex_cost_b
total_cost_per_mw = quadratic_cost_per_mw + linear_cost_per_mw

daily_gen_cost = total_cost_per_mw * avg_output * 24
daily_fixed_cost = ex_fixed * 24
daily_reserve_cost = reserve_cost_per_mw * 150 * 24  # 假设150MW备用

print(f"假设条件:")
print(f"  • Pmax: {ex_pmax} MW")
print(f"  • cost_a: {ex_cost_a}")
print(f"  • cost_b: {ex_cost_b}")
print(f"  • fixed: {ex_fixed}")
print(f"  • 平均出力: {avg_output} MW (60%负荷率)")
print(f"  • 系统总备用: 150 MW")

print(f"\n日运营成本:")
print(f"  • 发电成本: ${daily_gen_cost:,.0f}")
print(f"  • 固定成本: ${daily_fixed_cost:,.0f}")
print(f"  • 备用成本: ${daily_reserve_cost:,.0f}")
print(f"  • 合计: ${daily_gen_cost + daily_fixed_cost + daily_reserve_cost:,.0f}/天")

print(f"\n与EENS=1MWh的比较:")
print(f"  • 1MWh失负荷成本: $50")
print(f"  • 日平均失负荷成本(若EENS=1): $50/24 = ${50/24:.2f}/小时")
print(f"  • 保持可靠性的成本更高，所以EENS=0是合理的")

print("\n" + "=" * 70)
