"""
直接诊断：查看优化中的Load_Shed和Wind_Curt值
"""

import sys
sys.path.insert(0, 'D:\\pythoncode\\电价预测文件夹')

from 论文第二部分测试 import IEEERTS79System, ScenarioBasedUC

system = IEEERTS79System()
suc = ScenarioBasedUC(system, n_scenarios=3, two_stage=False)

print("【求解SUC1模型】")
results = suc.solve()

if results['status'] == 'Optimal':
    print(f"\n【检查优化变量的值】")
    
    # 从model中获取变量
    model = results['model']
    
    # 查找Load_Shed变量
    load_shed_vars = {v.name: v for v in model.variables() if 'Load_Shed' in v.name}
    wind_curt_vars = {v.name: v for v in model.variables() if 'Wind_Curt' in v.name}
    
    print(f"Load_Shed变量数：{len(load_shed_vars)}")
    print(f"Wind_Curt变量数：{len(wind_curt_vars)}")
    
    # 统计非零值
    nonzero_load_shed = sum(1 for v in load_shed_vars.values() if (v.varValue or 0) > 0.1)
    nonzero_wind_curt = sum(1 for v in wind_curt_vars.values() if (v.varValue or 0) > 0.1)
    
    print(f"非零Load_Shed数量：{nonzero_load_shed}/{len(load_shed_vars)}")
    print(f"非零Wind_Curt数量：{nonzero_wind_curt}/{len(wind_curt_vars)}")
    
    # 显示非零的Load_Shed
    if nonzero_load_shed > 0:
        print(f"\n【非零的Load_Shed】")
        for name in sorted(load_shed_vars.keys())[:10]:  # 前10个
            val = load_shed_vars[name].varValue or 0
            if val > 0.1:
                print(f"  {name}: {val:.2f} MW")
    
    # 显示非零的Wind_Curt
    if nonzero_wind_curt > 0:
        print(f"\n【非零的Wind_Curt】")
        for name in sorted(wind_curt_vars.keys())[:10]:  # 前10个
            val = wind_curt_vars[name].varValue or 0
            if val > 0.1:
                print(f"  {name}: {val:.2f} MW")
    
    print(f"\n【结论】")
    if nonzero_load_shed == 0 and nonzero_wind_curt == 0:
        print("  ✗ Load_Shed和Wind_Curt都被优化到0")
        print("  这就是为什么evaluate也显示0")
    else:
        print(f"  有{nonzero_load_shed}个Load_Shed和{nonzero_wind_curt}个Wind_Curt非零")
        print("  问题在evaluate逻辑，而不在优化本身")
        
else:
    print(f"✗ 求解失败：{results['status']}")
