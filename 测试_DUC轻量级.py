"""
轻量级DUC模型求解测试
用于快速验证成本修复效果
"""

import sys
sys.path.insert(0, 'D:\\pythoncode\\电价预测文件夹')

from 论文第二部分测试 import IEEERTS79System, DeterministicUC
import time

print("=" * 80)
print("DUC模型求解测试（成本修复验证）")
print("=" * 80)

try:
    # 加载系统
    print("\n【加载系统】")
    system = IEEERTS79System()
    
    # 创建DUC模型
    print("【创建DUC模型】")
    duc = DeterministicUC(system)
    
    # 求解
    print("【求解DUC模型】")
    print("  提示：求解可能需要5-15分钟...") 
    
    start = time.time()
    results = duc.solve()
    solve_time = time.time() - start
    
    # 显示结果
    print("\n【求解结果】")
    print(f"  状态: {results['status']}")
    print(f"  求解时间: {solve_time:.2f} 秒")
    
    if results['status'] == 'Optimal':
        total_cost_k = results['objective'] / 1000
        print(f"  总成本: {results['objective']:,.2f} $ = {total_cost_k:.2f} 千$")
        
        # 对比论文
        paper_value = 1183.5764
        error_pct = abs(total_cost_k - paper_value) / paper_value * 100
        
        print(f"\n【与论文对比】")
        print(f"  论文DUC成本: {paper_value:.2f} 千$")
        print(f"  计算结果:   {total_cost_k:.2f} 千$")
        print(f"  相对误差:   {error_pct:.2f}%")
        
        if error_pct < 10:
            print(f"  [OK] 误差<10%，成本修复基本成功！")
        elif error_pct < 20:
            print(f"  [WARN] 误差10-20%，成本接近但还需微调")
        else:
            print(f"  [FAIL] 误差>20%，成本修复不成功")
    else:
        print("  [FAIL] 求解未成功")
        
except Exception as e:
    print(f"[ERROR] 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n【下一步】")
print("如果DUC成本已接近论文值，继续修复：")
print("1. 实现蒙特卡洛评估")
print("2. 恢复CCV分段线性约束")
print("3. 添加支路潮流风险约束")
