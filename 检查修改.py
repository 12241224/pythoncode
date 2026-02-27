"""
检查修改是否成功 - 极简版本
"""

import sys
sys.path.insert(0, r'd:\pythoncode')

# 读取文件检查theta_ens值
with open(r'd:\pythoncode\论文第二部分测试.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
print("="*70)
print("修改检验 - 检查参数是否已更新")
print("="*70)

print("\n【检查项目1: theta_ens参数】")
if "self.theta_ens = 100" in content:
    print("✓ 已成功修改: self.theta_ens = 100 (原为50)")
else:
    print("✗ 未找到修改: self.theta_ens = 100")
    if "self.theta_ens = 50" in content:
        print("  仍为原值: self.theta_ens = 50")

print("\n【检查项目2: wind_std参数】")
if "error_std_1 = 0.25 * forecast[0]" in content:
    print("✓ 已成功修改: wind_std = 0.25 (原为0.15)")
else:
    print("✗ 未找到修改: wind_std = 0.25")
    if "error_std_1 = 0.15 * forecast[0]" in content:
        print("  仍为原值: wind_std = 0.15")

print("\n" + "="*70)
print("检验完成")
print("="*70)
print("""
如果两项都显示✓，说明修改成功！

下一步:
  1. 运行: python 论文第二部分测试.py
  2. 查看SUC1的输出:
     - 成本(应増至>800k$)
     - EENS(应増至>0.1MWh)
     - 弃风(应増至>1MWh)
""")
