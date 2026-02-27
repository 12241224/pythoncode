#!/usr/bin/env python3
"""
向量化优化版本 - 蒙特卡洛评估加速
将三重循环改为NumPy向量运算，预期10倍加速
"""

import numpy as np
import time as time_module

def calculate_branch_overflow_original(P_plan, system, critical_hours, n_wind_samples=2000):
    """原始的三重循环版本 (76 样本/秒)"""
    
    total_overflow_sum = 0.0
    overflow_count = 0
    total_evaluations = 0
    
    print("【原始三重循环版】")
    t1 = time_module.time()
    
    for t in critical_hours:
        wind_forecast = system.get_wind_forecast(t)
        
        for sample_idx in range(n_wind_samples):
            wind_scenario = []
            for w in range(len(system.wind_farms)):
                forecast = wind_forecast[w] if w < len(wind_forecast) else 0
                std = forecast * 0.15
                wind_power = np.random.normal(forecast, std)
                wind_power = np.clip(wind_power, 0, system.wind_farms['capacity'][w])
                wind_scenario.append(wind_power)
            
            # 对每条支路计算潮流
            for l in range(system.N_BRANCH):
                capacity = system.branches['capacity'][l]
                
                flow = 0.0
                # 发电机贡献
                for i in range(len(system.generators['bus'])):
                    try:
                        gen_bus = int(system.generators['bus'][i]) - 1
                        if gen_bus < system.N_BUS - 1:
                            P_val = P_plan[(i, t)].varValue
                            if P_val is not None:
                                flow += system.ptdf_matrix[l, gen_bus] * P_val
                    except:
                        pass
                
                # 风电贡献
                for w in range(len(system.wind_farms)):
                    try:
                        wind_bus = int(system.wind_farms['bus'][w]) - 1
                        if wind_bus < system.N_BUS - 1:
                            flow += system.ptdf_matrix[l, wind_bus] * wind_scenario[w]
                    except:
                        pass
                
                # 负荷贡献
                for bus in range(system.N_BUS - 1):
                    try:
                        flow -= system.ptdf_matrix[l, bus] * system.load_profile[t][bus]
                    except:
                        pass
                
                # 检查越限
                overflow = 0.0
                if abs(flow) > capacity:
                    overflow = abs(flow) - capacity
                    overflow_count += 1
                
                total_overflow_sum += overflow
                total_evaluations += 1
            
            # 进度显示
            if (sample_idx + 1) % 500 == 0:
                t_now = time_module.time()
                elapsed = t_now - t1
                rate = (sample_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  • {sample_idx+1} 样本: {elapsed:.2f}秒, {rate:.0f} 样本/秒")
    
    t2 = time_module.time()
    elapsed = t2 - t1
    
    print(f"\n【原始版性能】")
    print(f"  耗时: {elapsed:.2f}秒")
    print(f"  吞吐: {total_evaluations/elapsed:.0f} 潮流计算/秒")
    print(f"  样本率: {n_wind_samples*len(critical_hours)/elapsed:.0f} 样本/秒\n")
    
    return total_overflow_sum, overflow_count, total_evaluations


def calculate_branch_overflow_vectorized(P_plan, system, critical_hours, n_wind_samples=2000):
    """向量化版本 (预期 800+ 样本/秒)"""
    
    total_overflow_sum = 0.0
    overflow_count = 0
    total_evaluations = 0
    
    print("【向量化版本】")
    t1 = time_module.time()
    
    # 预处理: 准备P矩阵 (32×24)
    N_GEN = len(system.generators['bus'])
    N_TIME = system.T
    P_matrix = np.zeros((N_GEN, N_TIME))
    
    try:
        for i in range(N_GEN):
            for t in range(N_TIME):
                if (i, t) in P_plan:
                    P_val = P_plan[(i, t)].varValue
                    if P_val is not None:
                        P_matrix[i, t] = P_val
    except:
        pass
    
    # 预处理: 准备负荷矩阵 (24×24)
    Load_matrix = np.zeros((system.N_BUS-1, N_TIME))
    try:
        for bus in range(system.N_BUS - 1):
            for t in range(N_TIME):
                Load_matrix[bus, t] = system.load_profile[t][bus]
    except:
        pass
    
    # 预处理: 生成所有 n_wind_samples × 2 风电场景 (样本×风电场)
    np.random.seed(42)
    all_wind_scenarios = {}
    for t in critical_hours:
        wind_forecast = system.get_wind_forecast(t)
        wind_scenarios_t = []
        
        for sample_idx in range(n_wind_samples):
            wind_scenario = np.zeros(len(system.wind_farms))
            for w in range(len(system.wind_farms)):
                forecast = wind_forecast[w] if w < len(wind_forecast) else 0
                std = forecast * 0.15
                wind_power = np.random.normal(forecast, std)
                wind_power = np.clip(wind_power, 0, system.wind_farms['capacity'][w])
                wind_scenario[w] = wind_power
            
            wind_scenarios_t.append(wind_scenario)
        
        all_wind_scenarios[t] = np.array(wind_scenarios_t)  # (2000, 2)
    
    # 向量化计算: 对所有支路、时段、样本 (38×13×2000)
    # 而不是嵌套循环
    
    t_preprocess = time_module.time()
    print(f"  預处理耗时: {t_preprocess - t1:.2f}秒")
    
    # 发电机贡献: ptdf_matrix (38×23) × P_matrix (23×24) = (38×24)
    try:
        # 重新索引PTDF以适应发电机总线
        gen_buses = []
        for i in range(N_GEN):
            try:
                gen_bus = int(system.generators['bus'][i]) - 1
                gen_buses.append(gen_bus)
            except:
                gen_buses.append(-1)
        
        gen_buses = np.array(gen_buses)
        valid_gen = gen_buses >= 0
        valid_gen_idx = np.where(valid_gen)[0]
        
        # 只用有效的发电机
        P_valid = P_matrix[valid_gen_idx, :]  # (M, 24)
        ptdf_valid = system.ptdf_matrix[:, gen_buses[valid_gen_idx]]  # (38, M)
        
        # 矩阵乘法: (38, M) × (M, 24) = (38, 24)
        P_contribution = np.dot(ptdf_valid, P_valid)  # 蒙特卡洛-无关，仅时间维度
    except Exception as e:
        print(f"  警告: 发电机贡献计算失败 {e}")
        P_contribution = np.zeros((system.N_BRANCH, N_TIME))
    
    t_p_calc = time_module.time()
    
    # 主循环: 对所有样本进行向量化计算
    for t in critical_hours:
        # 负荷贡献: ptdf_matrix (38×23) × Load_matrix[t] (23,) = (38,)
        try:
            Load_contribution = np.dot(system.ptdf_matrix, Load_matrix[:, t])  # (38,)
        except:
            Load_contribution = np.zeros(system.N_BRANCH)
        
        # 获取该时刻所有样本的风电 (2000, 2)
        wind_samples_t = all_wind_scenarios[t]
        
        # 风电贡献: 对每个样本计算
        # ptdf_matrix (38×23) × wind_scenario (2,) → 需要扩展为 (38, 2000)
        try:
            wind_buses = []
            for w in range(len(system.wind_farms)):
                try:
                    wind_bus = int(system.wind_farms['bus'][w]) - 1
                    wind_buses.append(wind_bus)
                except:
                    wind_buses.append(-1)
            
            wind_buses = np.array(wind_buses)
            valid_wind = wind_buses >= 0
            valid_wind_idx = np.where(valid_wind)[0]
            
            ptdf_wind = system.ptdf_matrix[:, wind_buses[valid_wind_idx]]  # (38, K)
            
            # wind_samples_t (2000, K) → 需要转置计算
            # 为每个样本计算: (38, K) × (K, 1) = (38, 1)
            # 所有样本: (2000, 38)
            Wind_contribution = np.dot(wind_samples_t[:, valid_wind_idx], ptdf_wind.T).T  # (38, 2000)
        except Exception as e:
            print(f"  警告: 风电贡献计算失败 {e}")
            Wind_contribution = np.zeros((system.N_BRANCH, n_wind_samples))
        
        # 总潮流: (38, 2000)
        # 对每条支路: flow = P_contribution[:, t] (38) + Wind_contribution (38, 2000) - Load_contribution (38)
        P_contrib_t = P_contribution[:, t:t+1]  # (38, 1)
        Load_contrib_t = Load_contribution[:, np.newaxis]  # (38, 1)
        
        flow_all = P_contrib_t + Wind_contribution - Load_contrib_t  # (38, 2000)
        
        # 检查越限
        capacity = np.array(system.branches['capacity'])[:, np.newaxis]  # (38, 1)
        overflow = np.maximum(np.abs(flow_all) - capacity, 0)  # (38, 2000)
        
        overflow_count += np.sum(np.abs(flow_all) > capacity)
        total_overflow_sum += np.sum(overflow)
        total_evaluations += system.N_BRANCH * n_wind_samples
        
        # 进度显示
        if critical_hours.index(t) > 0 and critical_hours.index(t) % 3 == 0:
            t_now = time_module.time()
            elapsed = t_now - t1
            samples_done = (critical_hours.index(t) + 1) * n_wind_samples
            rate = samples_done / (elapsed - (t_p_calc - t1)) if elapsed > 0 else 0
            print(f"  • 时段{critical_hours.index(t)+1}: {elapsed:.2f}秒, {rate:.0f} 样本/秒")
    
    t2 = time_module.time()
    elapsed = t2 - t1
    
    print(f"\n【向量化版性能】")
    print(f"  耗时: {elapsed:.2f}秒")
    print(f"  吞吐: {total_evaluations/elapsed:.0f} 潮流计算/秒")
    print(f"  样本率: {n_wind_samples*len(critical_hours)/elapsed:.0f} 样本/秒")
    print(f"  【加速比: {26.0/elapsed:.1f}倍】\n")
    
    return total_overflow_sum, overflow_count, total_evaluations


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  蒙特卡洛评估向量化优化演示                                 ║
    ║  原始版 vs 向量化版性能对比                                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("说明:")
    print("  原始版: 6层嵌套循环 (样本→时段→支路→发电机/风电/负荷)")
    print("  向量化: NumPy矩阵运算，减少Python循环开销")
    print("  预期加速: 10倍 (76 → 800+ 样本/秒)")
    print()
    
    print("代码示例对比:")
    print()
    print("【原始版】")
    print("""
    for sample in 2000:
        for t in critical_hours:
            for l in 38:
                flow = sum(ptdf[l,i]*P[i,t] for i in 32)  # 32次乘法
                flow += sum(ptdf[l,w]*wind[w] for w in 2)   # 2次乘法
                flow -= sum(ptdf[l,b]*load[b] for b in 24)  # 24次乘法
                ...
    """)
    
    print("【向量化版】")
    print("""
    # 优化矩阵乘法
    P_contrib = ptdf @ P_matrix  # (38, 24)
    
    for t in critical_hours:
        for sample in 2000:
            flow_vec = P_contrib[:, t] + wind_contrib - load_contrib
            overflow = relu(abs(flow_vec) - capacity)
    """)
    
    print("\n" + "="*70)
    print("性能数据:")
    print("="*70)
    print()
    
    print("当前实际运行结果 (从程序输出):")
    print("  原始版 (DUC): 26秒, 2000样本×13时段 = 26年样本")
    print("             = 77 样本/秒")
    print()
    
    print("向量化版预期:")
    print("  目标: 2.6秒, 2000样本×13时段 = 26000样本")
    print("      = 770 样本/秒 (10倍加速)")
    print()
    
    print("如需实现:")
    print("  1. 修改 _calculate_branch_overflow_metrics 方法")
    print("  2. 预处理 P_matrix 和 Load_matrix (CPU时间)") 
    print("  3. 批量生成风电样本") 
    print("  4. 使用 np.dot() 替代 for 循环")
    print()
