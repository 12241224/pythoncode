"""
快速验证成本参数已修改
"""

import sys
sys.path.insert(0, 'D:\\pythoncode\\电价预测文件夹')

from 论文第二部分测试 import IEEERTS79System

system = IEEERTS79System()

print("=" * 80)
print("验证成本参数修改")
print("=" * 80)

# 显示各类型发电机的投资成本
print("\n【发电机固定成本汇总】")
total_fixed_cost_per_hour = 0

for gen_type in ['nuclear', 'coal_large', 'gas_medium', 'gas_large', 'hydro_small']:
    gens = system.generators[system.generators['type'] == gen_type]
    if len(gens) > 0:
        cost_c = gens['cost_c'].iloc[0]
        count = len(gens)
        total_c = cost_c * count
        total_fixed_cost_per_hour += total_c
        print(f"  {gen_type:15} ({count:2}台): cost_c={cost_c:6} $/小时 × {count} = {total_c:8} $/小时")

print(f"\n  合计固定成本: {total_fixed_cost_per_hour:8} $/小时")
print(f"  24小时固定成本: {total_fixed_cost_per_hour * 24:8} $")
print(f"  占总成本比例（112万$的约42%）: {total_fixed_cost_per_hour * 24 / 1180000 * 100:.1f}%")

print("\n【预期效果】")
print(f"  固定成本/24小时 = {total_fixed_cost_per_hour * 24:,.0f} $")
print(f"  如果燃料和维护成本约58万$，总成本应约 {(total_fixed_cost_per_hour * 24 + 580000)/1000:.1f} 千$")
print(f"  论文值为 1182-1184 千$")
