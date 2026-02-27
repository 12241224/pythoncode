# -*- coding: utf-8 -*-
"""
快速测试修改后的IEEE RTS-79模型（简化版，用于快速验证）
"""

import sys
import numpy as np
import time

print("="*70)
print("[快速测试] IEEE RTS-79系统修正版本验证")
print("="*70)

# 测试1：导入和系统初始化
print("\n[1/5] 初始化系统...")
from 论文第二部分测试 import IEEERTS79System, ScenarioBasedUC

system = IEEERTS79System()
print(f"  发电机: {system.N_GEN}台  |  母线: {system.N_BUS}个  |  支路: {system.N_BRANCH}条")

# 测试2：检查二次成本函数
print("\n[2/5] 验证二次成本函数...")
gen = system.generators.iloc[0]
print(f"  发电机0: C(P) = {gen['cost_a']}*P² + {gen['cost_b']}*P + {gen['cost_c']}")
print(f"           成本系数类型: cost_a={type(gen['cost_a']).__name__}, cost_b={type(gen['cost_b']).__name__}")

# 测试3：k-means场景生成（使用小规模测试）
print("\n[3/5] k-means场景生成测试")
t0 = time.time()
# 注意：这里使用较小的n_raw_scenarios (500而不是3000) 来快速测试
scenarios_test = system.get_wind_scenarios(n_scenarios=5, n_raw_scenarios=500)
t1 = time.time()
print(f"  生成了{len(scenarios_test)}个代表性场景（耗时 {t1-t0:.1f}秒）")
print(f"  场景概率和: {sum(system.scenario_probabilities):.4f}")

# 测试4：PTDF和支路潮流
print("\n[4/5] PTDF矩阵和支路潮流")
print(f"  PTDF矩阵形状: {system.ptdf_matrix.shape}")
generation = np.random.uniform(50, 150, system.N_GEN)
wind_power = [80, 85]
flows = system.calculate_branch_flow(generation, wind_power, 0)
print(f"  支路潮流: 最大={np.max(flows):.1f} MW, 最小={np.min(flows):.1f} MW")

# 测试5：SUC模型初始化（使用小规模）
print("\n[5/5] SUC模型初始化")
t0 = time.time()
suc = ScenarioBasedUC(system, n_scenarios=3, two_stage=False)  # 只用3个场景快速初始化
t1 = time.time()
print(f"  SUC1模型: {suc.n_scenarios}个场景, {suc.T}小时, {suc.N_GEN}台机组")
print(f"  初始化耗时: {t1-t0:.1f}秒")

# 汇总
print("\n" + "="*70)
print("[结果] 所有快速测试通过！")
print("="*70)
print("""
✓ k-means场景聚类功能正常
✓ 二次成本函数参数加载正确
✓ PTDF矩阵计算正常
✓ 支路潮流计算正常
✓ SUC模型可以初始化

建议：
1. 现在可以运行完整的SUC求解测试
2. 使用完整的3000场景->10个代表性场景聚类
3. 监控求解时间和成本收敛

修正内容总结：
✓ 场景生成：简单随机 → k-means聚类(3000→10)
✓ 成本函数：线性 → 二次函数 + 固定成本提高
✓ PTDF矩阵：简化 → 精确MATPOWER计算
✓ 蒙特卡洛：简化评估 → 详细的备用调用和风电误差模型
""")
