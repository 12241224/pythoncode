#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试改进修复的效果
- 测试RiskBasedUC中的theta_ens修改（从10000改为50）
- 测试DeterministicUC中的二次成本函数
- 测试场景数增加到10个
"""

import sys
import time

# 导入原始模块
sys.path.insert(0, r'd:\pythoncode')

print("=" * 60)
print("测试改进修复")
print("=" * 60)

# 导入必要的模块
from 论文第二部分测试 import IEEERTS79System, DeterministicUC, RiskBasedUC, ScenarioBasedUC

# 初始化系统
print("\n1. 初始化IEEE RTS-79系统...")
system = IEEERTS79System()
print(f"   系统规模: {system.N_GEN}台机组, {system.N_BUS}条母线, {system.N_BRANCH}条支路")
print(f"   风电场数: {len(system.wind_farms)}")

# 测试1: DUC模型（测试二次成本函数）
print("\n2. 运行DeterministicUC模型（测试二次成本函数）...")
print("   " + "-" * 50)
start_time = time.time()
try:
    duc = DeterministicUC(system)
    duc_results = duc.solve()
    duc_time = time.time() - start_time
    
    if duc_results and duc_results['status'] == 'Optimal':
        duc_metrics = duc.calculate_metrics(duc_results)
        print(f"   求解状态: {duc_results['status']}")
        print(f"   求解时间: {duc_time:.2f}秒")
        print(f"   总成本: ${duc_metrics['total_cost']:.2f}k")
        print(f"   EENS: {duc_metrics.get('EENS', 0):.4f} MWh")
        print(f"   弃风: {duc_metrics.get('wind_curtailment', 0):.4f} MWh")
    else:
        print(f"   求解失败: {duc_results['status'] if duc_results else '未知'}")
except Exception as e:
    print(f"   错误: {str(e)[:100]}")
    import traceback
    traceback.print_exc()

# 测试2: RUC模型（测试theta_ens修改）
print("\n3. 运行RiskBasedUC模型（测试theta_ens = 50）...")
print("   " + "-" * 50)
start_time = time.time()
try:
    ruc = RiskBasedUC(system, n_segments=10)
    
    # 验证theta_ens是否为50
    print(f"   验证: theta_ens = {ruc.theta_ens} (应为50)")
    print(f"   验证: theta_wind = {ruc.theta_wind} (应为50)")
    
    ruc_results = ruc.solve()
    ruc_time = time.time() - start_time
    
    if ruc_results and ruc_results['status'] == 'Optimal':
        ruc_metrics = ruc.calculate_metrics(ruc_results)
        print(f"   求解状态: {ruc_results['status']}")
        print(f"   求解时间: {ruc_time:.2f}秒")
        print(f"   总成本: ${ruc_metrics['total_cost']:.2f}k")
        print(f"   EENS: {ruc_metrics.get('EENS', 0):.4f} MWh")
        print(f"   弃风: {ruc_metrics.get('wind_curtailment', 0):.4f} MWh")
    else:
        print(f"   求解失败: {ruc_results['status'] if ruc_results else '未知'}")
except Exception as e:
    print(f"   错误: {str(e)[:100]}")
    import traceback
    traceback.print_exc()

# 测试3: SUC1模型（快速测试3个场景）
print("\n4. 运行ScenarioBasedUC模型（快速验证10个场景配置）...")
print("   " + "-" * 50)
try:
    # 首先验证n_scenarios是否正确设置
    suc_test = ScenarioBasedUC(system, n_scenarios=10, two_stage=False)
    print(f"   验证: 场景数 = {suc_test.n_scenarios} (应为10)")
    print(f"   验证: theta_ens = {suc_test.theta_ens} (应为50)")
    print(f"   验证: theta_wind = {suc_test.theta_wind} (应为50)")
    print(f"   建议: 完整运行可能需要10分钟（10个场景）")
    print(f"   下一步: 运行'论文第二部分测试.py'的main()函数进行完整测试")
except Exception as e:
    print(f"   错误: {str(e)[:100]}")

print("\n" + "=" * 60)
print("快速测试完成!")
print("=" * 60)
print("\n修改总结:")
print("  ✓ 1. theta_ens: 10000 → 50 (RiskBasedUC)")
print("  ✓ 2. DUC成本函数: 线性 → 二次分段线性")
print("  ✓ 3. 场景数: 3 → 10")
print("  ✓ 4. 时间限制: 300秒 → 600秒")
print("  ✓ 5. 间隙容差: 5% → 1%")
print("\n预期改进:")
print("  • DUC成本应显著增加（缺少二次成本）")
print("  • RUC成本应接近1183k$（theta_ens修复）")
print("  • 统计量（EENS、弃风）更加稳定（10个场景）")
