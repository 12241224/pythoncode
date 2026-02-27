# -*- coding: utf-8 -*-
"""
完整测试改进后的代码，包括指标计算
"""
import sys
sys.path.insert(0, '.')
from 论文第二部分测试 import IEEERTS79System, DeterministicUC

system = IEEERTS79System()

print('='*60)
print('完整测试：DUC 求解和指标计算')
print('='*60)

try:
    duc = DeterministicUC(system)
    print('[1/2] 开始求解 DUC...')
    duc_results = duc.solve()
    print('[1/2] 求解完成，状态: {}'.format(duc_results['status']))
    
    if duc_results['status'] == 'Optimal':
        print('[2/2] 开始计算指标（100个蒙特卡洛样本）...')
        duc_metrics = duc.calculate_metrics(duc_results)
        print('[2/2] 指标计算完成')
        if duc_metrics:
            print('\n指标摘要：')
            for key, value in sorted(duc_metrics.items())[:10]:  # 显示前10个指标
                print('  {}: {:.4f}'.format(key, value if isinstance(value, (int, float)) else 0))
        else:
            print('[x] 指标计算返回空结果')
    else:
        print('[x] DUC 求解不成功，无法计算指标')
except Exception as e:
    print('[x] 错误: {}: {}'.format(type(e).__name__, e))
    import traceback
    traceback.print_exc()

print('\n' + '='*60)
print('测试完成')
print('='*60)
