"""
测试修复成本函数后的结果
"""

import sys
sys.path.insert(0, 'D:\\pythoncode\\电价预测文件夹')

from 论文第二部分测试 import IEEERTS79System, DeterministicUC, RiskBasedUC

print("=" * 80)
print("测试修复成本函数后的结果")
print("=" * 80)

system = IEEERTS79System()

# 测试DUC模型
print("\n【DUC模型】")
duc = DeterministicUC(system)
duc_results = duc.solve()
duc_metrics = duc.calculate_metrics(duc_results)

print(f"  状态: {duc_results['status']}")
print(f"  总成本: {duc_results['objective']:,.2f} $ = {duc_results['objective']/1000:.2f} 千$")
print(f"  与论文对比: 论文DUC = 1183.5764 千$")
print(f"  误差: {abs(duc_results['objective']/1000 - 1183.5764) / 1183.5764 * 100:.2f}%")

# 测试RUC模型
print("\n【RUC模型】")
ruc = RiskBasedUC(system)
ruc_results = ruc.solve()
ruc_metrics = ruc.calculate_metrics(ruc_results)

print(f"  状态: {ruc_results['status']}")
print(f"  总成本: {ruc_results['objective']:,.2f} $ = {ruc_results['objective']/1000:.2f} 千$")
print(f"  与论文对比: 论文RUC = 1182.5962 千$")
print(f"  误差: {abs(ruc_results['objective']/1000 - 1182.5962) / 1182.5962 * 100:.2f}%")

print("\n【成本修复效果】")
if abs(duc_results['objective']/1000 - 1183) < 100:
    print("  ✓ DUC成本已接近论文值（在±100千$以内）")
else:
    print(f"  ✗ DUC成本还有较大偏差")

if abs(ruc_results['objective']/1000 - 1182) < 100:
    print("  ✓ RUC成本已接近论文值（在±100千$以内）")
else:
    print(f"  ✗ RUC成本还有较大偏差")
