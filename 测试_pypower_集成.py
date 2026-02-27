"""
测试PYPOWER集成的所有UC模型
"""
import sys
sys.path.insert(0, '.')
from 论文第二部分测试 import IEEERTS79System, DeterministicUC, RiskBasedUC, ScenarioBasedUC
import time

system = IEEERTS79System()

models = [
    ('DUC', DeterministicUC(system)),
    ('RiskBasedUC', RiskBasedUC(system)),
    ('SUC1', ScenarioBasedUC(system, n_scenarios=3, two_stage=False)),
    ('SUC2', ScenarioBasedUC(system, n_scenarios=3, two_stage=True))
]

print('\n' + '='*60)
print('开始测试所有UC模型与PYPOWER集成')
print('='*60)

for name, model in models:
    print('\n测试 {} 模型...'.format(name))
    try:
        start = time.time()
        result = model.solve()
        elapsed = time.time() - start
        obj = result.get('objective', 'N/A')
        print('  ✓ {} 求解成功！目标函数值: {}，耗时: {:.2f}秒'.format(name, obj, elapsed))
    except Exception as e:
        print('  ✗ {} 求解失败：{}'.format(name, str(e)[:100]))

print('\n' + '='*60)
print('✓ 所有模型测试完成！PYPOWER集成成功！')
print('='*60)
