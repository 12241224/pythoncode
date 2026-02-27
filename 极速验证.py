"""
极速验证 - 直接从源代码检查改进
不需要运行任何模型，仅读取源代码文件
"""

import re

print("\n" + "="*80)
print("方案A+B改进 - 代码验证 (10秒完成)")
print("="*80 + "\n")

# 读取源代码
with open(r'd:\pythoncode\论文第二部分测试.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 方案A: 检查theta_ens
print("【方案A: 失负荷成本权重】")
print("-" * 40)

# 查找ScenarioBasedUC中的theta_ens (在__init__方法中，应在1300行附近)
match_a = re.search(r'class ScenarioBasedUC:.*?self\.theta_ens = (\d+)', code, re.DOTALL)
if match_a:
    theta_value = int(match_a.group(1))
    print(f"代码中的 theta_ens = {theta_value}")
    
    if theta_value == 100:
        print("✓✓✓ 改进成功! (50 → 100)")
        status_a = True
    else:
        print(f"✗ 值为 {theta_value}，应为 100")
        status_a = False
else:
    print("✗ 无法找到theta_ens定义")
    status_a = False

# 方案B: 检查wind_std
print("\n【方案B: 风电不确定性】")
print("-" * 40)

# 查找error_std的系数
match_b = re.search(r'error_std_1 = ([\d.]+) \* forecast\[0\]', code)
if match_b:
    wind_factor = float(match_b.group(1))
    print(f"代码中的 wind_std系数 = {wind_factor:.2f}")
    
    if wind_factor == 0.25:
        print("✓✓✓ 改进成功! (0.15 → 0.25)")
        status_b = True
    else:
        print(f"✗ 系数为 {wind_factor:.2f}，应为 0.25")
        status_b = False
else:
    print("✗ 无法找到wind_std定义")
    status_b = False

# 总结
print("\n" + "="*80)
print("验证总结")
print("="*80 + "\n")

print(f"{'方案A (theta_ens)':40s} : {'✓ 已应用' if status_a else '✗ 未应用'}")
print(f"{'方案B (wind_std)':40s} : {'✓ 已应用' if status_b else '✗ 未应用'}")

if status_a and status_b:
    print("\n" + "🎉 "*10)
    print("\n    所有改进已成功应用到代码中!")
    print("\n预期改进效果:")
    print("  ✓ EENS: 0 → 0.2-0.5 MWh")
    print("  ✓ 成本: $716k → $850-1000k") 
    print("  ✓ 弃风: 0 → 5-15 MWh")
    print("\n立即运行模型验证效果:")
    print("  python 论文第二部分测试.py")
    
else:
    print("\n⚠ 部分改进可能未成功应用，请检查")

print("\n" + "="*80 + "\n")
