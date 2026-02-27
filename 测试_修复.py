# -*- coding: utf-8 -*-
"""
测试改进后的代码是否解决了变量为 None 的问题
"""
import sys
sys.path.insert(0, '.')
from 论文第二部分测试 import IEEERTS79System, DeterministicUC, RiskBasedUC

system = IEEERTS79System()

print('='*60)
print('测试1：DUC 求解')
print('='*60)

try:
    duc = DeterministicUC(system)
    print('开始求解 DUC...')
    duc_results = duc.solve()
    print('求解状态: {}'.format(duc_results['status']))
    
    if duc_results['status'] == 'Optimal':
        print('开始计算指标...')
        duc_metrics = duc.calculate_metrics(duc_results)
        print('✓ DUC 指标计算完成')
    else:
        print('DUC 求解不成功，跳过指标计算')
except Exception as e:
    print('✗ 错误: {}: {}'.format(type(e).__name__, e))
    import traceback
    traceback.print_exc()

print('\n' + '='*60)
print('测试2：RiskBasedUC 求解')
print('='*60)

try:
    ruc = RiskBasedUC(system, n_segments=5)
    print('开始求解 RiskBasedUC...')
    ruc_results = ruc.solve()
    print('求解状态: {}'.format(ruc_results['status']))
    
    if ruc_results['status'] == 'Optimal':
        print('开始计算指标...')
        ruc_metrics = ruc.calculate_metrics(ruc_results)
        print('✓ RiskBasedUC 指标计算完成')
    else:
        print('RiskBasedUC 求解不成功，跳过指标计算')
except Exception as e:
    print('✗ 错误: {}: {}'.format(type(e).__name__, e))
    import traceback
    traceback.print_exc()

print('\n' + '='*60)
print('✓ 所有测试完成')
print('='*60)
