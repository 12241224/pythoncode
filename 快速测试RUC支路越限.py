#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速测试RUC是否有支路越限风险"""

import sys
sys.path.insert(0, 'd:\\pythoncode')

from 论文第二部分测试 import IEEERTS79System, RiskBasedUC

print("=" * 60)
print("快速测试：RUC支路越限修复")
print("=" * 60)

# 加载系统
print("\n[1] 加载IEEE RTS-79系统...")
system = IEEERTS79System()

# 求解RUC
print("\n[2] 求解RUC模型...")
ruck = RiskBasedUC(system, n_segments=10)
results = ruck.solve()

# 计算指标
print("\n[3] 计算性能指标...")
metrics = ruck.calculate_metrics(results)

# 显示结果
print("\n" + "=" * 60)
print("RUC性能指标")
print("=" * 60)
print(f"求解状态: {results['status']}")
print(f"总成本: ¥{results['objective']/1000:.2f}k")
print(f"\n可靠性指标:")
print(f"  失负荷概率: {metrics['prob_load_shedding']:.4%}")
print(f"  平均EENS: {metrics['expected_eens']:.6f} MWh/h")
print(f"  弃风概率: {metrics['prob_wind_curtailment']:.4%}")
print(f"  平均弃风: {metrics['expected_wind_curtailment']:.6f} MWh/h")
print(f"\n支路越限指标:")
print(f"  越限概率: {metrics['prob_overflow']:.4%}")
print(f"  平均越限: {metrics['expected_overflow']:.6f} MW/h")

# 检查R_flow是否有值
print(f"\n支路越限风险详情:")
total_r_flow = 0
for t in range(24):
    for l in range(system.N_BRANCH):
        r_flow = results['R_flow'][(t, l)].varValue or 0
        total_r_flow += r_flow
        if r_flow > 0.01:
            print(f"  时段{t:2d}, 支路{l:2d}: {r_flow:8.4f} MW")

if total_r_flow > 0.1:
    print(f"\n✓ RUC有支路越限风险！总风险: {total_r_flow:.4f} MW-h")
else:
    print(f"\n✗ RUC仍无支路越限风险 (总计: {total_r_flow:.6f} MW-h)")

print("\n" + "=" * 60)
