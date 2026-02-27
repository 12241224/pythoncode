"""
快速测试改进后的SUC模型效果
运行单个SUC1模型，对比改进前后的成本/EENS
"""

import sys
sys.path.insert(0, r'd:\pythoncode')
import warnings
warnings.filterwarnings('ignore')
import time

# 导入系统和模型类
print("加载系统和模型类...")

# 从论文脚本中导入关键组件
import numpy as np
import pandas as pd
import pulp
from scipy import stats
from scipy.integrate import quad
from sklearn.cluster import KMeans
from pypower.api import case24_ieee_rts

# ======================================================================
# 重复定义SUC类和IEEE RTS系统（简化版，仅为演示）
# ======================================================================

# 从论文第二部分测试.py复制系统定义
exec("""
import numpy as np
import pandas as pd
import pulp
import time
from scipy import stats
from sklearn.cluster import KMeans
from pypower.api import case24_ieee_rts

RTS79_GENERATORS = [
    {'type': 'nuclear', 'count': 2, 'Pmin': 50, 'Pmax': 400, 
     'cost_a': 0.005, 'cost_b': 15.0, 'cost_c': 750,
     'ramp_up': 100, 'ramp_down': 100, 'min_up': 8, 'min_down': 8,
     'startup_cost': 1500, 'buses': [22, 23]},
    
    {'type': 'coal_large', 'count': 7, 'Pmin': 150, 'Pmax': 197, 
     'cost_a': 0.006, 'cost_b': 14.0, 'cost_c': 500,
     'ramp_up': 80, 'ramp_down': 80, 'min_up': 8, 'min_down': 8,
     'startup_cost': 1200, 'buses': [1, 2, 7, 13, 15, 16, 23]},
    
    {'type': 'gas_medium', 'count': 6, 'Pmin': 20, 'Pmax': 100, 
     'cost_a': 0.008, 'cost_b': 17.0, 'cost_c': 250,
     'ramp_up': 50, 'ramp_down': 50, 'min_up': 4, 'min_down': 4,
     'startup_cost': 500, 'buses': [1, 2, 3, 5, 8, 9]},
    
    {'type': 'gas_large', 'count': 5, 'Pmin': 50, 'Pmax': 400, 
     'cost_a': 0.007, 'cost_b': 18.0, 'cost_c': 300,
     'ramp_up': 120, 'ramp_down': 120, 'min_up': 5, 'min_down': 5,
     'startup_cost': 900, 'buses': [6, 10, 12, 14, 18]},
    
    {'type': 'hydro_small', 'count': 12, 'Pmin': 0, 'Pmax': 50, 
     'cost_a': 0.0, 'cost_b': 20.0, 'cost_c': 100,
     'ramp_up': 50, 'ramp_down': 50, 'min_up': 1, 'min_down': 1,
     'startup_cost': 400, 'buses': list(range(1, 25))[:12]},
]

# 生成完整发电机列表
generators = []
for gen_type in RTS79_GENERATORS:
    count = gen_type.pop('count', 1)
    for i in range(count):
        gen = dict(gen_type)
        generators.append(gen)
print(f"[OK] 定义了{len(generators)}台发电机")
""")

# 现在导入实际的IEEE RTS系统和模型
from 论文第二部分测试 import IEEERTS79System, ScenarioBasedUC

print("="*80)
print("SUC模型改进效果验证")
print("="*80)

# 创建系统
print("\n【系统初始化】")
system = IEEERTS79System(year=2019)
print(f"✓ 系统创建成功:发电机{system.N_GEN}台,母线{system.N_BUS}个,风电{len(system.wind_farms)}场")

# 创建并求解SUC1
print("\n【求解SUC1 (单阶段,5场景)】")
print("-"*80)
suc1 = ScenarioBasedUC(system, n_scenarios=5, two_stage=False)

print(f"当前参数设置:")
print(f"  theta_ens = {suc1.theta_ens} $/MWh (改进前为50)")
print(f"  theta_wind = {suc1.theta_wind} $/MWh")
print(f"  风电不确定性 = 25% (改进前为15%)")
print(f"\n求解模型...")

try:
    results1 = suc1.solve()
    
    if suc1.model.status == 1:  # 最优
        cost1 = suc1.model.objective.value()
        eens1 = results1.get('EENS', 0)
        wind1 = results1.get('Wind_Curtailment', 0)
        
        print(f"\n✓ 求解成功!")
        print(f"\n【SUC1结果】(改进方案A+B)")
        print(f"  成本: ${cost1:,.2f} (改进前: $716,289)")
        print(f"  EENS: {eens1:.4f} MWh (改进前: 0.0000,目标: 0.23-0.40)")
        print(f"  弃风: {wind1:.4f} MWh (改进前: 0.0000)")
        print(f"  失负荷概率: {results1.get('Loss_Shedding_Probability', 0):.2%}")
        
        # 成本差异分析
        cost_gain = cost1 - 716289.43
        cost_gain_percent = (cost_gain / 716289.43) * 100
        
        print(f"\n【改进效果分析】")
        print(f"  成本增加: ${cost_gain:,.2f} ({cost_gain_percent:+.1f}%)")
        
        # EENS改进分析
        if eens1 > 0.001:
            print(f"  ✓ EENS已从0增加到{eens1:.4f} MWh")
            eens_target = 0.32  # 论文目标0.23-0.40的中点
            eens_error = abs(eens1 - eens_target) / eens_target * 100
            print(f"  ✓ 与论文目标偏差: {eens_error:.1f}%")
        else:
            print(f"  ⚠ EENS仍为0,可能需要进一步调整参数")
        
        print(f"\n【对标论文Table IV】")
        print(f"  目标成本: ~1184 k$")
        print(f"  当前成本: {cost1/1000:.0f} k$ (相对偏差: {((cost1-1184000)/1184000)*100:.1f}%)")
        
        print(f"\n【建议】")
        if eens1 > 0.1:
            print(f"  ✓✓ 改进方案A+B效果良好!")
            print(f"     EENS已增加,成本也有所增加")
            if cost1 > 900000:
                print(f"     成本已接近论文目标范围")
        else:
            print(f"  ⚠ EENS增长不足,考虑:")
            print(f"    - 继续增加theta_ens(当前100,可尝试150-200)")
            print(f"    - 再增加风电不确定性(当前25%,可尝试30%)")
            
    else:
        print(f"✗ 求解失败,状态: {suc1.model.status}")
        
except Exception as e:
    print(f"✗ 错误: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成")
print("="*80)
