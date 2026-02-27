"""
用不同scenarios数量测试SUC
"""

import sys
sys.path.insert(0, 'D:\\pythoncode\\电价预测文件夹')

from 论文第二部分测试 import IEEERTS79System, ScenarioBasedUC

system = IEEERTS79System()

print("="*60)
print("SUC模型 - 不同scenarios数量对比")
print("="*60)

for n_scen in [3, 5, 10]:
    print(f"\n【{n_scen}个scenarios】")
    suc = ScenarioBasedUC(system, n_scenarios=n_scen, two_stage=False)
    results = suc.solve()
    
    if results['status'] == 'Optimal':
        metrics = suc.calculate_metrics(results)
        print(f"  总成本：{metrics['total_cost']:.0f}千$")
        print(f"  EENS：{metrics['expected_eens']:.2f} MWh（期望5.2）")
        print(f"  弃风：{metrics['expected_wind_curtailment']:.2f} MWh（期望15.1）")
    else:
        print(f"  求解失败：{results['status']}")

print("\n" + "="*60)
print("论文期望值（SUC1）:")
print("  总成本：850千$")
print("  EENS：5.2 MWh")
print("  弃风：15.1 MWh")
