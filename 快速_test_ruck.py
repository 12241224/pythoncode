#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速测试RUC参数影响"""

import sys
sys.path.insert(0, 'd:\pythoncode')

# 导入主程序的关键类和函数
exec(open('d:\pythoncode\论文第二部分测试.py').read().split('if __name__')[0])

# 只运行DUC和RUC
print("开始快速测试RUC...")
system = IEEESystem()
print("[OK] 系统加载完成")

duc = DeterministicUC(system)
duc_results = duc.solve()
duc_metrics = duc.calculate_metrics(duc_results)

print("\nDUC成本:", duc_metrics['total_cost'], "EENS:", duc_metrics['expected_eens'])

ruc = RiskBasedUC(system, n_segments=10)
ruc_results = ruc.solve()
ruc_metrics = ruc.calculate_metrics(ruc_results)

print("RUC成本:", ruc_metrics['total_cost'])
print("RUC EENS:", ruc_metrics['expected_eens'])
print("RUC 弃风:", ruc_metrics['expected_wind_curtailment'])

# 打印R_ens和R_wind原值
print("\n原始R值:")
for t in range(5):
    r_ens = ruc_results['R_ens'][t].varValue
    r_wind = ruc_results['R_wind'][t].varValue
    print(f"  时段{t}: R_ens={r_ens:.2f}, R_wind={r_wind:.2f}")
