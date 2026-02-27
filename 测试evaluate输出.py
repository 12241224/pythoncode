"""
测试修改后的evaluate是否正确输出EENS
"""

import sys
sys.path.insert(0, 'D:\\pythoncode\\电价预测文件夹')

from 论文第二部分测试 import IEEERTS79System, ScenarioBasedUC

system = IEEERTS79System()
suc = ScenarioBasedUC(system, n_scenarios=3, two_stage=False)

print("【求解SUC1模型】")
results = suc.solve()

if results['status'] == 'Optimal':
    print(f"✓ 求解成功")
    print(f"  总成本：${results['objective']:,.0f}")
    
    print(f"\n【计算指标】")
    try:
        metrics = suc.calculate_metrics(results)
        print(f"✓ calculate_metrics 成功")
        print(f"  total_cost: {metrics['total_cost']:.0f}千$")
        print(f"  prob_load_shedding: {metrics['prob_load_shedding']:.4f}")
        print(f"  expected_eens: {metrics['expected_eens']:.2f} MWh")
        print(f"  prob_wind_curt: {metrics['prob_wind_curtailment']:.4f}")
        print(f"  expected_wind_curt: {metrics['expected_wind_curtailment']:.2f} MWh")
    except Exception as e:
        print(f"✗ calculate_metrics 失败:")
        print(f"  {type(e).__name__}: {str(e)[:200]}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ 求解失败：{results['status']}")
