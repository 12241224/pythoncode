#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUC模型修复验证脚本
验证以下关键修复：
1. 惩罚系数：1000/100 → 50（正确的风险成本）
2. 二次成本函数：平均斜率 → 最大斜率（更准确）
3. 失负荷上界：max_load → max_load*0.1（更合理）
4. calculate_metrics：确定性 → 蒙特卡洛评估（更准确）
"""

import sys
import time
sys.path.insert(0, r'd:\pythoncode')

print("=" * 70)
print("SUC模型修复效果验证")
print("=" * 70)

print("\n【修复清单】")
print("-" * 70)
print("""
1. ✓ 失负荷/弃风惩罚系数: 1000/100 → 50 (theta_ens/theta_wind)
   影响: 允许优化器在成本和失电风险之间权衡，而不是极力避免失电

2. ✓ 二次成本函数近似: 平均斜率 → 最大斜率
   影响: 更准确的成本估计，避免低估发电成本

3. ✓ Load_Shed上界: max_load → max_load*0.1  
   影响: 限制最多允许失负荷10%（更现实）

4. ✓ calculate_metrics改进: 确定性 → 蒙特卡洛(1000样本)
   影响: 更准确的后验EENS和弃风评估
""")

try:
    from 论文第二部分测试 import IEEERTS79System, ScenarioBasedUC
    
    print("\n【初始化系统】")
    print("-" * 70)
    system = IEEERTS79System()
    print(f"✓ IEEE RTS-79系统已加载:")
    print(f"  • 发电机: {system.N_GEN}台")     
    print(f"  • 母线: {system.N_BUS}个")
    print(f"  • 支路: {system.N_BRANCH}条")
    print(f"  • 风电场: {len(system.wind_farms)}个")
    
    # 快速测试（3场景）
    print("\n【快速验证 - SUC1 (3场景)】")
    print("-" * 70)
    start_time = time.time()
    suc1_quick = ScenarioBasedUC(system, n_scenarios=3, two_stage=False)
    
    print(f"验证参数:")
    print(f"  • theta_ens = {suc1_quick.theta_ens} (应为50)")
    print(f"  • theta_wind = {suc1_quick.theta_wind} (应为50)")
    print(f"  • n_scenarios = {suc1_quick.n_scenarios} (应为3)")
    
    print(f"\n求解中...")
    suc1_results = suc1_quick.solve()
    suc1_time = time.time() - start_time
    
    if suc1_results and suc1_results['status'] == 'Optimal':
        print(f"✓ 求解成功!")
        print(f"  • 求解时间: {suc1_time:.2f}秒")
        print(f"  • 目标函数: ${suc1_results['objective']:.0f}")
        
        suc1_metrics = suc1_quick.calculate_metrics(suc1_results)
        
        print(f"\n【SUC1结果】")
        print(f"  • 总成本: ${suc1_metrics['total_cost']:.2f}k")
        print(f"  • EENS: {suc1_metrics['expected_eens']:.4f} MWh")
        print(f"  • 弃风: {suc1_metrics['expected_wind_curtailment']:.4f} MWh")
        print(f"  • 失负荷概率: {suc1_metrics['prob_load_shedding']:.2%}")
        print(f"  • 弃风概率: {suc1_metrics['prob_wind_curtailment']:.2%}")
    else:
        print(f"✗ 求解失败: {suc1_results['status'] if suc1_results else '未知'}")
    
    # 验证修复效果
    print(f"\n【修复效果验证】")
    print("-" * 70)
    
    # 预期改进：
    # 1. 成本应该更高（因为增加了真实的二次成本和允许失负荷）
    # 2. EENS应该 > 0（因为降低了惩罚系数）
    # 3. 弃风也应该 > 0
    
    if suc1_metrics['total_cost'] and suc1_metrics['total_cost'] > 700:
        print("✓ 成本合理: 超过700k$（说明二次成本被计入）")
    else:
        print("⚠ 成本仍较低: 检查二次成本计算")
    
    if suc1_metrics['expected_eens'] > 0:
        print(f"✓ EENS > 0: {suc1_metrics['expected_eens']:.4f} MWh（失负荷被适当量化）")
    else:
        print("✗ EENS仍为0: 需要进一步调试")
    
    if suc1_metrics['expected_wind_curtailment'] > 0:
        print(f"✓ 弃风 > 0: {suc1_metrics['expected_wind_curtailment']:.4f} MWh")
    else:
        print("⚠ 弃风为0: 可能正常（取决于负荷和风电）")
    
    # 对标论文目标
    print(f"\n【对标论文Table IV】")
    print(f"  • 论文SUC1成本目标: ~1184k$")
    print(f"  • 论文EENS目标: 0.23-0.40 MWh")
    
    if suc1_metrics['total_cost']:
        cost_deviation = abs(suc1_metrics['total_cost'] - 1184) / 1184 * 100
        print(f"  • 当前成本与目标偏差: {cost_deviation:.1f}%")
        
        if cost_deviation < 30:
            print(f"    ✓ 偏差较小（< 30%）")
        else:
            print(f"    ⚠ 偏差较大（≥ 30%）")
    
    print("\n" + "=" * 70)
    print("快速验证完成！")
    print("=" * 70)
    
    print("\n【下一步建议】")
    print("-" * 70)
    print("1. 如果EENS和成本仍不满意，运行:")
    print("   python 论文第二部分测试.py")
    print("   检查所有四个模型（DUC/RUC/SUC1/SUC2）的完整结果")
    print()
    print("2. 如果成本偏差>30%，检查:")
    print("   • cost_a系数是否正确")
    print("   • startup_cost参数是否合理")
    print("   • 发电机参数是否与论文一致")
    print()
    print("3. 如果EENS仍为0，可能需要:")
    print("   • 增加蒙特卡洛样本数")
    print("   • 或者直接读取优化结果中的Load_Shed值")
    
except Exception as e:
    print(f"✗ 错误: {str(e)}")
    import traceback
    traceback.print_exc()
