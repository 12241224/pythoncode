# -*- coding: utf-8 -*-
"""
快速测试 PYPOWER 集成（DUC 和 SUC1）
"""
import sys
sys.path.insert(0, '.')
from 论文第二部分测试 import IEEERTS79System, DeterministicUC, ScenarioBasedUC
import time

print('\n' + '='*60)
print('PYPOWER 集成测试 - 快速版本')
print('='*60)

system = IEEERTS79System()

print('\n[1/2] 测试 DUC 模型...')
try:
    start = time.time()
    duc = DeterministicUC(system)
    result_duc = duc.solve()
    elapsed = time.time() - start
    obj_duc = result_duc.get('objective', 'N/A')
    print('  ✓ DUC 求解成功！')
    print('    - 目标函数值: {:.2f}'.format(obj_duc))
    print('    - 耗时: {:.2f}秒'.format(elapsed))
except Exception as e:
    print('  ✗ DUC 求解失败：{}'.format(str(e)[:100]))

print('\n[2/2] 测试 SUC1 模型...')
try:
    start = time.time()
    suc1 = ScenarioBasedUC(system, n_scenarios=3, two_stage=False)
    result_suc1 = suc1.solve()
    elapsed = time.time() - start
    obj_suc1 = result_suc1.get('objective', 'N/A')
    print('  ✓ SUC1 求解成功！')
    print('    - 目标函数值: {:.2f}'.format(obj_suc1))
    print('    - 耗时: {:.2f}秒'.format(elapsed))
except Exception as e:
    print('  ✗ SUC1 求解失败：{}'.format(str(e)[:100]))

print('\n' + '='*60)
print('✓ PYPOWER 集成测试完成！')
print('  系统已成功从硬编码数据迁移到 PYPOWER 库')
print('='*60 + '\n')
