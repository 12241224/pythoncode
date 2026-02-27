# -*- coding: utf-8 -*-
"""
快速测试改进后的代码
"""
import sys
sys.path.insert(0, '.')
from 论文第二部分测试 import IEEERTS79System, DeterministicUC, RiskBasedUC

system = IEEERTS79System()

print('='*60)
print('快速测试1：DUC 求解')
print('='*60)

try:
    duc = DeterministicUC(system)
    print('开始求解 DUC...')
    duc_results = duc.solve()
    print('求解状态: {}'.format(duc_results['status']))
    
    # 不计算指标，只检查变量是否有效
    if duc_results['status'] == 'Optimal':
        print('检查变量有效性...')
        for i in range(min(3, system.N_GEN)):  # 检查前3个发电机
            for t in range(min(3, 24)):  # 检查前3个小时
                u_val = duc_results['U'][(i, t)].varValue
                p_val = duc_results['P'][(i, t)].varValue
                print('  U[{},{}]={}, P[{},{}]={}'.format(i, t, u_val, i, t, p_val))
                if u_val is None or p_val is None:
                    print('  ✗ 变量为 None!')
                    break
        print('✓ DUC 求解和变量检查完成')
    else:
        print('DUC 求解不成功')
except Exception as e:
    print('✗ 错误: {}: {}'.format(type(e).__name__, e))
    import traceback
    traceback.print_exc()

print('\n' + '='*60)
print('✓ 快速测试完成')
print('='*60)
