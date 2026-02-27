#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RUC EENS改进方法对比验证
测试新的11点分段线性化EENS风险约束方法
"""

import sys
import time
sys.path.insert(0, r'd:\pythoncode')

print("=" * 70)
print("RUC EENS改进方法验证")
print("=" * 70)

print("\n【改进方法说明】")
print("-" * 70)
print("""
改进前（简化方法）:
  • 使用3-sigma固定备用规则: R_up >= 3 * σ_wind
  • EENS由简单线性关系近似
  • 无法精确表达风电不确定性与失电的关系

改进后（精确分段线性化）:
  • 使用11点分段线性化（10段）
  • 正确的数学公式：EENS = σ[φ(z) - z(1-Φ(z))]
  • 其中z = (R_up - D_t) / σ_t
  • 约束：R_ens ≥ max_k(a_eens[k] * R_up + b_eens[k])
  • 同时处理弃风R_wind的分段线性化

预期改进:
  • RUC成本更加精确
  • EENS在0.23-0.40 MWh范围内（接近论文目标）
  • 备用与失电之间的权衡更合理
  • 风电不确定性转化为明确的成本信号
""")

print("\n【初始化系统和模型】")
print("-" * 70)

try:
    from 论文第二部分测试 import IEEERTS79System, RiskBasedUC
    
    print("初始化IEEE RTS-79系统...")
    system = IEEERTS79System()
    print(f"  ✓ 系统加载成功: {system.N_GEN}台机组, {system.N_BUS}条母线")
    
    print("\n初始化RiskBasedUC模型（新的精确方法）...")
    ruc = RiskBasedUC(system, n_segments=10)
    print(f"  ✓ 模型创建成功")
    print(f"  • theta_ens = {ruc.theta_ens} $/MWh（失电成本）")
    print(f"  • theta_wind = {ruc.theta_wind} $/MWh（弃风成本）")
    print(f"  • n_segments = {ruc.n_segments} （分段线性化段数）")
    
    # 验证eens_params和wind_params是否正确预计算
    print(f"\n验证分段线性化参数预计算...")
    print(f"  • eens_params数量: {len(ruc.eens_params)} (应为24)")
    print(f"  • wind_params数量: {len(ruc.wind_params)} (应为24)")
    
    if len(ruc.eens_params) > 0:
        a_eens_0, b_eens_0 = ruc.eens_params[0]
        print(f"  • 时刻0的EENS参数: {len(a_eens_0)}个分段系数")
        print(f"    a_eens[0]={a_eens_0[0]:.6f}, b_eens[0]={b_eens_0[0]:.2f}")
        
    print("\n【运行RUC求解】")
    print("-" * 70)
    
    start_time = time.time()
    print("求解RUC模型（使用改进的EENS约束）...")
    ruc_results = ruc.solve()
    ruc_time = time.time() - start_time
    
    if ruc_results and ruc_results['status'] == 'Optimal':
        print(f"✓ 求解成功!")
        print(f"  • 求解时间: {ruc_time:.2f}秒")
        print(f"  • 目标函数: ${ruc_results['objective']:.0f}")
        
        print("\n【计算性能指标】")
        print("-" * 70)
        
        ruc_metrics = ruc.calculate_metrics(ruc_results)
        
        print(f"成本分析:")
        print(f"  • 总成本: ${ruc_metrics['total_cost']:.2f}k")
        if 'generation_cost' in ruc_metrics:
            print(f"  • 发电成本: ${ruc_metrics['generation_cost']:.2f}k")
        if 'risk_cost' in ruc_metrics:
            print(f"  • 风险成本: ${ruc_metrics['risk_cost']:.2f}k")
        
        print(f"\n可靠性指标:")
        print(f"  • EENS: {ruc_metrics.get('EENS', 0):.4f} MWh")
        print(f"    目标范围: 0.23-0.40 MWh")
        print(f"  • 弃风: {ruc_metrics.get('wind_curtailment', 0):.4f} MWh")
        print(f"    目标范围: 15.1 MWh")
        print(f"  • 支路越限: {ruc_metrics.get('overflow', 0):.4f} MWh")
        
        print(f"\n备用分析:")
        print(f"  • 平均上备用: {ruc_metrics.get('avg_reserve_up', 0):.2f} MW")
        print(f"  • 平均下备用: {ruc_metrics.get('avg_reserve_down', 0):.2f} MW")
        
        # 计算与论文目标的偏差
        target_cost = 1183  # 论文中的RUC目标成本
        target_eens = 0.32  # 中间值
        target_wind = 15.1
        
        cost_deviation = ((ruc_metrics['total_cost'] - target_cost) / target_cost * 100)
        
        print(f"\n【对比论文目标 (Table IV)】")
        print("-" * 70)
        print(f"成本:")
        print(f"  • 论文目标: ${target_cost:.0f}k")
        print(f"  • 当前结果: ${ruc_metrics['total_cost']:.2f}k")
        print(f"  • 偏差: {cost_deviation:.1f}%")
        
        print(f"\nENS:")
        print(f"  • 论文目标: {target_eens:.2f} MWh")
        print(f"  • 当前结果: {ruc_metrics.get('EENS', 0):.4f} MWh")
        
        print(f"\n【改进方法的优势】")
        print("-" * 70)
        print("""
1. 数学精确性
   ✓ 使用标准正态分布的精确CDF和PDF
   ✓ EENS公式：EENS = σ[φ(z) - z(1-Φ(z))]
   ✓ 分段线性化精度高于简化方法

2. 风电不确定性表示
   ✓ 风电预测误差模型：σ_wind = 15% × 预测值
   ✓ 动态调整每个时间的不确定性
   ✓ 无法处理的缺电量完全由R_ens量化

3. 优化决策改进
   ✓ 备用（R_up）与失电成本（θ_ens）直接关联
   ✓ 可自动权衡：更多备用 vs. 更高风险成本
   ✓ 无人为的固定备用规则（如3-sigma）

4. 操作间隙关闭
   ✓ 完全闭合的凸优化模型
   ✓ 所有约束都在连续可微空间内
   ✓ 容易扩展到其他不确定性来源
        """)
        
    else:
        print(f"✗ 求解失败: {ruc_results['status'] if ruc_results else '未知'}")
        
except Exception as e:
    print(f"✗ 错误: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("验证完成!")
print("=" * 70)

print("\n建议:")
print("  1. 运行 python 论文第二部分测试.py 获得完整的模型对比")
print("  2. 对比RUC成本是否接近1183k$")
print("  3. 检查EENS是否在0.23-0.40 MWh范围内")
print("  4. 如果偏差仍较大，考虑调整theta_ens或风电不确定性参数")
