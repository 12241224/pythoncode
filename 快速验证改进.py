"""
快速验证SUC改进效果 - 简化版本
只测试基本功能，不涉及蒙特卡洛
"""

import sys
import os
sys.path.insert(0, r'd:\pythoncode')
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SUC改进方案验证 - 简化版")
print("="*80)

# 加载系统
exec(open(r'd:\pythoncode\论文第二部分测试.py').read().split('if __name__ == "__main__"')[0])

print("\n【验证改动是否生效】")
print("-"*80)

# 创建SUC模型
print("\n1. 创建SUC1模型（3个场景）...")
try:
    suc1 = ScenarioBasedUC(system, n_scenarios=3, two_stage=False)
    print(f"   ✓ theta_ens = {suc1.theta_ens} (应为100)")
    print(f"   ✓ theta_wind = {suc1.theta_wind} (应为50)")
    
    # 检查是否为100
    if suc1.theta_ens == 100:
        print("   ✓ 改进方案A已生效！")
    else:
        print("   ✗ 改进方案A未生效！")
        
except Exception as e:
    print(f"   ✗ 错误: {str(e)[:100]}")

print("\n2. 检查风电不确定性...")
try:
    # 查看场景中的风电波动
    scenario_winds = []
    for s in range(min(3, suc1.n_scenarios)):
        for t in range(min(3, suc1.T)):
            w_power = suc1.scenarios[s][t]
            scenario_winds.append(sum(w_power))
    
    wind_std = __import__('numpy').std(scenario_winds)
    wind_mean = __import__('numpy').mean(scenario_winds)
    wind_cv = wind_std / wind_mean if wind_mean > 0 else 0
    
    print(f"   场景风电均值: {wind_mean:.1f} MW")
    print(f"   场景风电标差: {wind_std:.1f} MW")
    print(f"   变异系数: {wind_cv:.1%} (应接近25%)")
    
    if wind_cv > 0.20:  # 至少20%的变异系数
        print("   ✓ 改进方案B已生效！(风电不确定性增加)")
    else:
        print("   ⚠ 风电不确定性可能未充分增加")
        
except Exception as e:
    print(f"   ✗ 错误: {str(e)[:100]}")

print("\n" + "="*80)
print("验证完成！")
print("="*80)
print("""
关键检查项:
  ✓ theta_ens should be 100 (was 50)
  ✓ wind uncertainty should be 25% (was 15%)
  
现在可以运行完整模型:
  python 论文第二部分测试.py
  
观察SUC1的输出:
  - 成本应该 > $800k (之前$716k)
  - EENS应该 > 0.1 MWh (之前0)
""")
