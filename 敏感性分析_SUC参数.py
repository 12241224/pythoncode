"""
SUC敏感性分析: 测试theta_ens和风电不确定性对EENS和成本的影响
目标: 找到合适的参数组合使EENS与论文目标0.23-0.40 MWh接近
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 确保可以导入论文模型
sys.path.insert(0, r'd:\pythoncode')

# ============================================================================
# 步骤1: 设置测试参数
# ============================================================================

test_configs = [
    # (theta_ens, wind_std_factor, description)
    (50, 0.15, "当前设置"),
    (75, 0.15, "略增theta"),
    (100, 0.15, "中等theta"),
    (150, 0.15, "高theta"),
    (200, 0.15, "很高theta"),
    
    (100, 0.20, "中风电+风不确定性增加"),
    (100, 0.25, "中风电+更高风不确定性"),
    (150, 0.20, "高theta+风不确定性增加"),
    (150, 0.25, "高theta+更高风不确定性"),
    (200, 0.25, "很高theta+很高风不确定性"),
]

print("="*100)
print("SUC模型敏感性分析 - 寻找合适的参数使EENS与论文目标一致")
print("="*100)
print(f"\n论文目标: EENS ≈ 0.23-0.40 MWh, 成本 ≈ 1184 k$")
print(f"当前状态: EENS = 0.0000 MWh, 成本 = 716 k$\n")

print("当前准备测试以下参数组合:")
for i, (theta, wind_std, desc) in enumerate(test_configs, 1):
    print(f"  {i:2d}. theta_ens={theta:3d} $/MWh, wind_std={wind_std:.2%}  ({desc})")

print("\n" + "="*100)

# ============================================================================
# 步骤2: 自动修改和运行论文第二部分测试.py
# ============================================================================

main_file = r'd:\pythoncode\论文第二部分测试.py'
results_data = []

print("\n开始敏感性分析，这可能需要几分钟... (共{}个配置)\n".format(len(test_configs)))

for config_idx, (theta_ens, wind_std_factor, desc) in enumerate(test_configs, 1):
    print(f"[{config_idx}/{len(test_configs)}] 测试: {desc} (theta={theta_ens}, wind_std={wind_std_factor:.1%})")
    
    # 读取原始文件
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 创建临时修改的版本
    modified_content = content
    
    # 修改theta_ens参数
    # 在ScenarioBasedUC.__init__中修改
    modified_content = modified_content.replace(
        "self.theta_ens = 50  # $/MWh",
        f"self.theta_ens = {theta_ens}  # $/MWh"
    )
    
    # 修改theta_wind参数
    modified_content = modified_content.replace(
        "self.theta_wind = 50  # $/MWh",
        f"self.theta_wind = {theta_ens}  # $/MWh (matching theta_ens)"
    )
    
    # 修改风电标准差倍数 - 在wind_samples生成处修改
    # 查找并替换风电场景生成的不确定性参数
    modified_content = modified_content.replace(
        "wind_std = total_wind_forecast * 0.15  # 15%的标准差",
        f"wind_std = total_wind_forecast * {wind_std_factor:.2f}  # {wind_std_factor:.1%}的标准差"
    )
    
    # 保存修改后的版本
    temp_file = r'd:\pythoncode\_temp_sensitivity_test.py'
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    # 运行修改后的文件，并捕获结果
    import subprocess
    try:
        result = subprocess.run(
            ['python', temp_file],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=r'd:\pythoncode'
        )
        
        output = result.stdout + result.stderr
        
        # 从输出中提取关键数据
        import re
        
        # 提取SUC1成本
        cost_match = re.search(r'总成本:\s*\$?([\d,]+\.?\d*)\s*k\$', output)
        cost = float(cost_match.group(1).replace(',', '')) if cost_match else None
        
        # 提取EENS
        eens_match = re.search(r'EENS:\s*([\d.]+)\s*MWh', output)
        eens = float(eens_match.group(1)) if eens_match else None
        
        # 提取弃风
        wind_match = re.search(r'弃风:\s*([\d.]+)\s*MWh', output)
        wind_curt = float(wind_match.group(1)) if wind_match else None
        
        if cost is not None and eens is not None:
            # 计算对论文目标的偏差
            cost_error = (cost - 1184) / 1184 * 100 if cost else None
            eens_from_target = eens - 0.32  # 论文目标中点
            
            result_dict = {
                'theta_ens': theta_ens,
                'wind_std': f"{wind_std_factor:.1%}",
                'description': desc,
                'cost': cost,
                'EENS': eens,
                'Wind_Curt': wind_curt,
                'cost_error%': cost_error,
                'eens_error': eens_from_target
            }
            
            results_data.append(result_dict)
            
            # 打印结果
            status = "✓" if 0.23 <= eens <= 0.40 else "✗"
            print(f"  {status} 成本: {cost:7.1f}k$ (+{cost_error:6.1f}%) | "
                  f"EENS: {eens:6.4f} MWh | 弃风: {wind_curt:6.4f} MWh")
        else:
            print(f"  ❌ 无法提取结果数据，可能运行失败")
            
    except subprocess.TimeoutExpired:
        print(f"  ⏱️  超时（>60秒）")
    except Exception as e:
        print(f"  ❌ 错误: {str(e)[:50]}")
    
    # 清理临时文件
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except:
            pass

print("\n" + "="*100)
print("敏感性分析结果汇总")
print("="*100)

if results_data:
    # 创建DataFrame
    df = pd.DataFrame(results_data)
    
    # 显示完整表格
    print("\n详细结果表:")
    print(df.to_string(index=False))
    
    # 找到最接近论文目标的配置
    print("\n【论文目标对标】")
    print(f"  目标EENS范围: 0.23-0.40 MWh")
    print(f"  目标成本: 1184 k$")
    
    # 找EENS最接近目标的配置
    target_eens = 0.32  # 0.23-0.40的中点
    target_cost = 1184
    
    df['eens_distance'] = np.abs(df['EENS'] - target_eens)
    df['cost_distance'] = np.abs(df['cost'] - target_cost)
    
    best_eens_idx = df['eens_distance'].idxmin()
    best_cost_idx = df['cost_distance'].idxmin()
    
    print(f"\n  ✓ 最接近目标EENS的配置:")
    best_eens_row = df.iloc[best_eens_idx]
    print(f"    theta_ens: {best_eens_row['theta_ens']}, wind_std: {best_eens_row['wind_std']}")
    print(f"    EENS: {best_eens_row['EENS']:.4f} (vs 目标 0.32)")
    print(f"    成本: {best_eens_row['cost']:.1f}k$ (vs 目标 1184)")
    
    print(f"\n  ✓ 最接近目标成本的配置:")
    best_cost_row = df.iloc[best_cost_idx]
    print(f"    theta_ens: {best_cost_row['theta_ens']}, wind_std: {best_cost_row['wind_std']}")
    print(f"    成本: {best_cost_row['cost']:.1f}k$ (vs 目标 1184)")
    print(f"    EENS: {best_cost_row['EENS']:.4f} (vs 目标 0.32)")
    
    # 绘制趋势分析
    print("\n【趋势分析】")
    print("\n  当theta_ens增加时的影响:")
    for theta in df['theta_ens'].unique():
        subset = df[df['theta_ens'] == theta]
        avg_eens = subset['EENS'].mean()
        avg_cost = subset['cost'].mean()
        print(f"    theta_ens={theta:3d}: 平均EENS={avg_eens:.4f} MWh, 平均成本={avg_cost:7.1f}k$")
    
    print("\n  当wind_std增加时的影响:")
    for wind_std in sorted(df['wind_std'].unique()):
        subset = df[df['wind_std'] == wind_std]
        avg_eens = subset['EENS'].mean()
        avg_cost = subset['cost'].mean()
        print(f"    wind_std={wind_std:>5s}: 平均EENS={avg_eens:.4f} MWh, 平均成本={avg_cost:7.1f}k$")
    
    # 保存结果到CSV
    output_file = r'd:\pythoncode\敏感性分析结果.csv'
    df[['theta_ens', 'wind_std', 'description', 'cost', 'EENS', 'Wind_Curt', 'cost_error%']].to_csv(
        output_file, index=False, encoding='utf-8-sig'
    )
    print(f"\n✓ 结果已保存到: {output_file}")
    
else:
    print("\n❌ 未能获得任何有效结果")

print("\n" + "="*100)
print("建议方案")
print("="*100)
print("""
根据上述敏感性分析结果:

1️⃣ 如果EENS仍全为0:
   • 增加theta_ens到200-300范围
   • 增加风电不确定性到20-25%
   • 这会让失负荷变成经济合理的选择

2️⃣ 如果找到接近论文目标的参数:
   • 使用该参数组合更新论文第二部分测试.py
   • 运行完整模型(DUC/RUC/SUC1/SUC2)进行验证

3️⃣ 如果成本仍偏低(如700-800k$ vs 1184k$):
   • 可能需要:
     a) 调整发电机参数(cost_a, cost_b)
     b) 增加约束条件(如最小启动时间)
     c) 改进风电场景生成方法

4️⃣ 下一步行动:
   • 根据上表选择最合适的参数组合
   • 在论文第二部分测试.py中永久更新这些参数
   • 运行完整模型进行最终验证
""")

print("\n敏感性分析完成！")
