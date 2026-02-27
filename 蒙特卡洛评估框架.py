"""
蒙特卡洛评估的实现框架
将添加到各UC模型中
"""

# ====== 蒙特卡洛评估函数（通用） ======

def evaluate_with_monte_carlo(self, results, n_samples=1000):
    """
    基于优化后的调度方案，用蒙特卡洛模拟评估失负荷和弃风风险
    
    参数：
    - results: UC求解结果，包含P, U, R_up, R_down等决策变量
    - n_samples: 蒙特卡洛样本数（默认1000）
    
    返回：
    - metrics: 包含失负荷概率、EENS、弃风概率、弃风期望等
    """
    
    total_eens = 0
    total_wind_curt = 0
    sample_load_shed = 0  # 发生失负荷的样本数
    sample_wind_curt = 0  # 发生弃风的样本数
    
    # 提取优化决策
    P = results['P']  # 发电功率
    R_up = results['R_up']  # 上备用
    R_down = results['R_down']  # 下备用
    
    print("  开始蒙特卡洛评估...")
    
    for sample_idx in range(n_samples):
        sample_eens = 0
        sample_wind_curt = 0
        sample_has_loadshed = False
        sample_has_windcurt = False
        
        for t in range(self.T):
            # 1. 获取优化决策和需求
            total_gen = sum(P[(i, t)].varValue if P[(i, t)].varValue else 0 
                          for i in range(self.N_GEN))
            total_r_up = sum(R_up[(i, t)].varValue if R_up[(i, t)].varValue else 0 
                           for i in range(self.N_GEN))
            total_r_down = sum(R_down[(i, t)].varValue if R_down[(i, t)].varValue else 0 
                            for i in range(self.N_GEN))
            
            load = sum(self.system.load_profile[t])
            wind_forecast = sum(self.system.get_wind_forecast(t))
            wind_std = wind_forecast * 0.15
            
            # 2. 生成随机风电场景（正态分布）
            import numpy as np
            wind_deviation = np.random.normal(0, wind_std)
            wind_actual = wind_forecast + wind_deviation
            wind_actual = max(0, min(wind_actual, sum(w['capacity'] for w in self.system.wind_farms)))
            
            # 3. 可以调用的备用（受爬坡率约束）
            # 简化：假设可以调用全部备用
            called_reserve = min(total_r_up, max(0, load - total_gen - wind_actual))
            
            # 4. 计算不平衡（失负荷或弃风）
            supply_with_reserve = total_gen + wind_actual + called_reserve
            
            if supply_with_reserve < load:
                # 缺电：失负荷
                eens = load - supply_with_reserve
                sample_eens += eens
                sample_has_loadshed = True
            elif total_gen + wind_actual > total_gen + total_r_down:
                # 逆向：可能弃风
                # 弃风 = max(0, 风电 - (可消纳风电))
                max_wind = load + total_r_down - total_gen  # 最多能消纳的风电
                if wind_actual > max_wind:
                    wind_curt = wind_actual - max_wind
                    sample_wind_curt += wind_curt
                    sample_has_windcurt = True
        
        # 统计样本结果
        total_eens += sample_eens
        total_wind_curt += sample_wind_curt
        
        if sample_has_loadshed:
            sample_load_shed += 1
        if sample_has_windcurt:
            sample_wind_curt += 1
        
        # 进度显示
        if (sample_idx + 1) % (n_samples // 10) == 0:
            print(f"    ...已完成 {sample_idx + 1}/{n_samples} 样本")
    
    # 计算平均指标
    prob_load_shedding = sample_load_shed / n_samples
    expected_eens = total_eens / n_samples / self.T
    
    prob_wind_curtailment = sample_wind_curt / n_samples
    expected_wind_curtailment = total_wind_curt / n_samples / self.T
    
    metrics = {
        'prob_load_shedding': prob_load_shedding,
        'expected_eens': expected_eens,
        'prob_wind_curtailment': prob_wind_curtailment,
        'expected_wind_curtailment': expected_wind_curtailment,
    }
    
    print(f"  完成蒙特卡洛评估:")
    print(f"    失负荷概率: {prob_load_shedding:.4f}")
    print(f"    期望EENS: {expected_eens:.4f} MWh/h")
    print(f"    弃风概率: {prob_wind_curtailment:.4f}")
    print(f"    期望弃风: {expected_wind_curtailment:.4f} MWh/h")
    
    return metrics


# ====== 修改calculate_metrics()的方式 ======
"""
将当前calculate_metrics()中的硬编码值替换为：

def calculate_metrics(self, results):
    # ... 获取成本等其他指标 ...
    
    # 用蒙特卡洛评估替代硬编码的metrics
    mc_metrics = self.evaluate_with_monte_carlo(results, n_samples=1000)
    
    metrics = {
        'total_cost': results['objective'] / 1000 if results['objective'] else None,
        'prob_load_shedding': mc_metrics['prob_load_shedding'],
        'expected_eens': mc_metrics['expected_eens'],
        'prob_wind_curtailment': mc_metrics['prob_wind_curtailment'],
        'expected_wind_curtailment': mc_metrics['expected_wind_curtailment'],
        'prob_overflow': 0.05,
        'expected_overflow': 0.2
    }
    
    return metrics
"""

print(__doc__)
