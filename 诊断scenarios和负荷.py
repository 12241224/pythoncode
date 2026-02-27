"""
诊断scenarios和负荷的关系
"""

import sys
sys.path.insert(0, 'D:\\pythoncode\\电价预测文件夹')

import numpy as np
from 论文第二部分测试 import IEEERTS79System, ScenarioBasedUC

system = IEEERTS79System()
suc = ScenarioBasedUC(system, n_scenarios=10, two_stage=False)

print("【负荷信息】")
loads = [sum(system.load_profile[t]) for t in range(24)]
print(f"  最小负荷：{min(loads):.0f} MW")
print(f"  平均负荷：{np.mean(loads):.0f} MW")
print(f"  最大负荷：{max(loads):.0f} MW")

print(f"\n【风电scenarios】")
for s in range(min(3, len(suc.scenarios))):
    winds = [sum(suc.scenarios[s][t]) for t in range(24)]
    print(f"\n  Scenario {s}:")
    print(f"    最小风电：{min(winds):.0f} MW")
    print(f"    平均风电：{np.mean(winds):.0f} MW")
    print(f"    最大风电：{max(winds):.0f} MW")

print(f"\n【关键统计】")
# 检查是否存在 Gen + Wind > Load 的情况
total_cap = sum(system.generators['Pmax'])
print(f"  总发电容量：{total_cap:.0f} MW")
print(f"  最大负荷：{max(loads):.0f} MW")
print(f"  容量-负荷差：{total_cap - max(loads):.0f} MW")

# 最大风电
max_wind_all = max([sum(suc.scenarios[s][t]) for s in range(len(suc.scenarios)) for t in range(24)])
print(f"  最大风电（所有scenarios）：{max_wind_all:.0f} MW")
print(f"  理论最大供电（发电+风电）：{total_cap + max_wind_all:.0f} MW")
print(f"  vs最大负荷：{max(loads):.0f} MW")
print(f"  过剩容量：{total_cap + max_wind_all - max(loads):.0f} MW")

print(f"\n【弃风产生的条件】")
print(f"  需要：P + W > Load + R_down")
print(f"  即：发电+风电 > 负荷+下备用")
print(f"  通常在最小负荷+最大风电时发生")

min_load = min(loads)
print(f"\n  最小负荷时刻：{min_load:.0f} MW")
print(f"  加上下备用（假设50MW）：{min_load + 50:.0f} MW")
print(f"  vs最大风电+可用发电：{max_wind_all}+ ~{total_cap - min_load:.0f} = {max_wind_all + total_cap - min_load:.0f}")
print(f"  应该有弃风产生！")
