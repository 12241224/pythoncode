import pandas as pd
import numpy as np

# 读取预测结果
results = pd.read_csv('d:\\pythoncode\\储能电价预测结果.csv')

# 基本统计
print("=" * 70)
print("储能电价预测结果 - 合理性分析")
print("=" * 70)

print("\n【1】基本数据统计")
print(f"预测周期: {len(results)} 小时 ({len(results)/24:.1f} 天)")
print(f"电价范围: {results['predicted_price'].min():.2f} - {results['predicted_price'].max():.2f} 元/MWh")
print(f"平均电价: {results['predicted_price'].mean():.2f} 元/MWh")
print(f"标准差: {results['predicted_price'].std():.2f} 元/MWh")
print(f"价格变异系数(CV): {results['predicted_price'].std()/results['predicted_price'].mean():.2%}")

# 时间特征分析
print("\n【2】时间特征分析 - 时间周期模式")
valley_prices = results[results['is_valley_hour']==True]['predicted_price']
morning_prices = results[results['is_morning_peak']==True]['predicted_price']
evening_prices = results[results['is_evening_peak']==True]['predicted_price']

print(f"\n谷底时段(0-5时) - 低负荷:")
print(f"  平均价格: {valley_prices.mean():.2f} 元/MWh")
print(f"  最低/最高: {valley_prices.min():.2f} / {valley_prices.max():.2f}")
print(f"  数据点: {len(valley_prices)}")

print(f"\n早高峰时段(8-11时) - 晨间需求高:")
print(f"  平均价格: {morning_prices.mean():.2f} 元/MWh")
print(f"  最低/最高: {morning_prices.min():.2f} / {morning_prices.max():.2f}")
print(f"  数据点: {len(morning_prices)}")

print(f"\n晚高峰时段(18-21时) - 晚间需求最高:")
print(f"  平均价格: {evening_prices.mean():.2f} 元/MWh")
print(f"  最低/最高: {evening_prices.min():.2f} / {evening_prices.max():.2f}")
print(f"  数据点: {len(evening_prices)}")

# 充放电决策分析
print("\n【3】充放电决策分析")
charge_count = (results['storage_action']=='charge').sum()
discharge_count = (results['storage_action']=='discharge').sum()
hold_count = (results['storage_action']=='hold').sum()

print(f"充电(charge): {charge_count} 小时 ({charge_count/len(results):.1%})")
print(f"放电(discharge): {discharge_count} 小时 ({discharge_count/len(results):.1%})")
print(f"保持(hold): {hold_count} 小时 ({hold_count/len(results):.1%})")

# 检查充放电价差
charge_times = results[results['storage_action']=='charge']
discharge_times = results[results['storage_action']=='discharge']

if len(charge_times) > 0 and len(discharge_times) > 0:
    print(f"\n充电时段平均价格: {charge_times['predicted_price'].mean():.2f} 元/MWh")
    print(f"放电时段平均价格: {discharge_times['predicted_price'].mean():.2f} 元/MWh")
    
    price_diff = discharge_times['predicted_price'].mean() - charge_times['predicted_price'].mean()
    print(f"平均价差: {price_diff:.2f} 元/MWh (利差: {(price_diff/charge_times['predicted_price'].mean()):.1%})")

# 分析时间顺序的合理性
print("\n【4】充放电时间顺序合理性检查")
charge_hours = results[results['storage_action']=='charge']['hour_ahead'].values
discharge_hours = results[results['storage_action']=='discharge']['hour_ahead'].values

print(f"充电时段小时数: {sorted(charge_hours.tolist())}")
print(f"放电时段小时数: {sorted(discharge_hours.tolist())}")

# 峰谷反向分析
print("\n【5】预测模式合理性检查")
morning_avg = results[results['hour_of_day'].isin([8,9,10,11])]['predicted_price'].mean()
evening_avg = results[results['hour_of_day'].isin([18,19,20,21])]['predicted_price'].mean()
valley_avg = results[results['hour_of_day'].isin([0,1,2,3,4,5])]['predicted_price'].mean()
midday_avg = results[results['hour_of_day'].isin([12,13,14,15,16,17])]['predicted_price'].mean()

print(f"谷底(0-5时)价格: {valley_avg:.2f} 元/MWh (最低)")
print(f"早高峰(8-11时)价格: {morning_avg:.2f} 元/MWh")
print(f"中午(12-17时)价格: {midday_avg:.2f} 元/MWh")
print(f"晚高峰(18-21时)价格: {evening_avg:.2f} 元/MWh (应最高)")

if morning_avg > valley_avg:
    print("✓ 早高峰价格高于谷底 (符合预期)")
else:
    print("✗ 早高峰价格低于谷底 (异常!)")
    
if evening_avg > valley_avg:
    print("✓ 晚高峰价格高于谷底 (符合预期)")
else:
    print("✗ 晚高峰价格低于谷底 (异常!)")

if evening_avg > morning_avg:
    print("✓ 晚高峰价格高于早高峰 (符合一般预期)")
else:
    print("△ 晚高峰价格不高于早高峰 (可能模式)")
    
print("\n【6】充放电时段设置问题分析")
# 晚高峰 - 应该放电，不应该充电
evening_charge = results[(results['is_evening_peak']==True) & (results['storage_action']=='charge')]
evening_discharge = results[(results['is_evening_peak']==True) & (results['storage_action']=='discharge')]
print(f"\n晚高峰(18-21时)分配:")
print(f"  充电时段: {len(evening_charge)} 小时   <- 问题: 不应该充电!")
print(f"  放电时段: {len(evening_discharge)} 小时  <- 正确: 应该放电")

if len(evening_charge) > 0:
    print(f"\n  ⚠️ 问题时段: {evening_charge['hour_ahead'].tolist()}")
    print(f"  价格数据: {evening_charge['predicted_price'].values}")
    print(f"  ⚠️ 这些时段在高电价时段应该放电而非充电!")

# 早高峰 - 应该放电，通常不充电
morning_charge = results[(results['is_morning_peak']==True) & (results['storage_action']=='charge')]
morning_discharge = results[(results['is_morning_peak']==True) & (results['storage_action']=='discharge')]
print(f"\n早高峰(8-11时)分配:")
print(f"  充电时段: {len(morning_charge)} 小时")
print(f"  放电时段: {len(morning_discharge)} 小时")

# 谷底 - 应该充电
valley_charge = results[(results['is_valley_hour']==True) & (results['storage_action']=='charge')]
valley_discharge = results[(results['is_valley_hour']==True) & (results['storage_action']=='discharge')]
print(f"\n谷底时段(0-5时)分配:")
print(f"  充电时段: {len(valley_charge)} 小时   <- 正确: 应该充电")
print(f"  放电时段: {len(valley_discharge)} 小时  <- 问题: 不应该放电")

if len(valley_discharge) > 0:
    print(f"  ⚠️ 问题: 在谷底(低电价)时段出现{len(valley_discharge)}小时的放电建议")

print("\n【7】利润计算示例")
storage_efficiency = 0.85
max_rate = 100

if len(discharge_times) > 0 and len(charge_times) > 0:
    min_price = charge_times['predicted_price'].min()
    max_price = discharge_times['predicted_price'].max()
    
    price_diff = max_price - min_price
    profit_per_cycle = max_rate * price_diff * storage_efficiency
    
    print(f"最低充电价格: {min_price:.2f} 元/MWh")
    print(f"最高放电价格: {max_price:.2f} 元/MWh") 
    print(f"最佳价差: {price_diff:.2f} 元/MWh")
    print(f"假设条件: 100MW容量 x 1小时 x {storage_efficiency:.0%}效率")
    print(f"单次循环利润: ￥{profit_per_cycle:.2f}")
    
    daily_cycles = min(charge_count, discharge_count)  # 一天最多循环次数
    if daily_cycles > 0:
        daily_profit = profit_per_cycle * daily_cycles
        print(f"如果每天{daily_cycles}个循环: ￥{daily_profit:.2f}/天")

print("\n" + "=" * 70)
print("【分析总结】")
print("=" * 70)

# 总体评估
issues = []
if len(evening_charge) > 0:
    issues.append(f"晚高峰时段出现充电建议({len(evening_charge)}小时)，应为放电")
if len(valley_discharge) > 0:
    issues.append(f"谷底时段出现放电建议({len(valley_discharge)}小时)，应为充电")
if evening_avg < valley_avg * 1.2:
    issues.append(f"晚高峰价格({evening_avg:.2f})与谷底价格({valley_avg:.2f})的差异不够大")

if issues:
    print("\n找到的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ 预测结果模式合理，充放电决策逻辑正确")

print("\n预测数据特征:")
print(f"  - 数据完整: {len(results)} 个小时")
print(f"  - 价格变化合理: CV={results['predicted_price'].std()/results['predicted_price'].mean():.2%}")
print(f"  - 峰谷差异: {(evening_avg - valley_avg):.2f} 元/MWh ({(evening_avg/valley_avg - 1):.1%})")
