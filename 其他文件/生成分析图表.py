import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 读取数据
results = pd.read_csv('d:\\pythoncode\\储能电价预测结果.csv')

# 创建大型图表
fig = plt.figure(figsize=(16, 12))

# ========== 图1: 电价按小时趋势 (主视图) ==========
ax1 = plt.subplot(3, 2, 1)
hours = results['hour_ahead'].values
prices = results['predicted_price'].values

ax1.plot(hours, prices, 'b-o', linewidth=2.5, markersize=6, label='预测电价')

# 标记早晚高峰
morning_peak_mask = results['is_morning_peak'] == True
evening_peak_mask = results['is_evening_peak'] == True
valley_mask = results['is_valley_hour'] == True

ax1.fill_between(hours[morning_peak_mask], 0, prices[morning_peak_mask], alpha=0.2, color='red', label='早高峰')
ax1.fill_between(hours[evening_peak_mask], 0, prices[evening_peak_mask], alpha=0.2, color='orange', label='晚高峰')
ax1.fill_between(hours[valley_mask], 0, prices[valley_mask], alpha=0.2, color='green', label='谷底')

# 标记充放电
charge_mask = results['storage_action'] == 'charge'
discharge_mask = results['storage_action'] == 'discharge'

ax1.scatter(hours[charge_mask], prices[charge_mask], s=150, marker='v', color='green', 
           edgecolors='darkgreen', linewidth=2, label='充电', zorder=5)
ax1.scatter(hours[discharge_mask], prices[discharge_mask], s=150, marker='^', color='red', 
           edgecolors='darkred', linewidth=2, label='放电', zorder=5)

ax1.axhline(y=88.57, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='早高峰均价(88.57)')
ax1.axhline(y=60.62, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='晚高峰均价(60.62)❌')
ax1.axhline(y=68.18, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='谷底均价(68.18)')

ax1.set_xlabel('预测小时', fontsize=11, fontweight='bold')
ax1.set_ylabel('电价 (元/MWh)', fontsize=11, fontweight='bold')
ax1.set_title('图1: 24小时电价预测 + 充放电决策', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc='best')
ax1.set_xticks(hours)

# ========== 图2: 各时段价格对比 ==========
ax2 = plt.subplot(3, 2, 2)
time_periods = ['谷底\n(0-5时)', '早高峰\n(8-11时)', '中午低谷\n(12-17时)', '晚高峰\n(18-21时)']
avg_prices = [68.18, 88.57, 60.66, 60.62]
colors_bar = ['green', 'red', 'blue', 'orange']

bars = ax2.bar(time_periods, avg_prices, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

# 添加值标签
for bar, price in zip(bars, avg_prices):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{price:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 标记问题
ax2.text(3, 60.62 + 3, '❌ 异常过低\n低于谷底!', ha='center', fontsize=9, 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax2.set_ylabel('平均电价 (元/MWh)', fontsize=11, fontweight='bold')
ax2.set_title('图2: 各时段平均电价对比 - 晚高峰异常', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 100])
ax2.grid(True, alpha=0.3, axis='y')

# ========== 图3: 充放电决策分布 ==========
ax3 = plt.subplot(3, 2, 3)
actions = ['充电\n(25%)\n6小时', '放电\n(29.2%)\n7小时', '保持\n(45.8%)\n11小时']
action_counts = [6, 7, 11]
action_colors = ['green', 'red', 'gray']

wedges, texts, autotexts = ax3.pie(action_counts, labels=actions, autopct='%1.1f%%',
                colors=action_colors, startangle=90, textprops={'fontsize': 10})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax3.set_title('图3: 充放电决策分布', fontsize=12, fontweight='bold')

# ========== 图4: 充放电时段价格对比 ==========
ax4 = plt.subplot(3, 2, 4)
segment_names = ['充电时段\n平均价格', '放电时段\n平均价格']
segment_prices = [51.18, 85.07]
segment_colors_bar = ['green', 'red']

bars2 = ax4.bar(segment_names, segment_prices, color=segment_colors_bar, alpha=0.7, 
               edgecolor='black', linewidth=1.5, width=0.5)

# 添加值标签和利润计算
for bar, price in zip(bars2, segment_prices):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{price:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 显示价差
price_diff = 85.07 - 51.18
ax4.text(0.5, 35, f'价差: {price_diff:.2f} 元/MWh\n利差: 66.2%', 
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax4.set_ylabel('平均电价 (元/MWh)', fontsize=11, fontweight='bold')
ax4.set_title('图4: 充放电价格对比 - 价差充足', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 100])
ax4.grid(True, alpha=0.3, axis='y')

# ========== 图5: 各小时充放电决策详情 ==========
ax5 = plt.subplot(3, 2, 5)
colors_action = {'charge': 'green', 'discharge': 'red', 'hold': 'gray'}
bar_colors = [colors_action[action] for action in results['storage_action']]

bars3 = ax5.bar(hours, prices, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.5)

# 标记问题时段
ax5.axvline(x=19, color='red', linestyle=':', linewidth=3, label='问题: 19时充电(应放电)')
ax5.text(19, 95, '❌', fontsize=16, ha='center')

ax5.set_xlabel('预测小时', fontsize=11, fontweight='bold')
ax5.set_ylabel('电价 (元/MWh)', fontsize=11, fontweight='bold')
ax5.set_title('图5: 小时级充放电决策 - 晚高峰问题突出', fontsize=12, fontweight='bold')
ax5.set_xticks(hours)
ax5.grid(True, alpha=0.3, axis='y')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', alpha=0.7, edgecolor='black', label='充电'),
    Patch(facecolor='red', alpha=0.7, edgecolor='black', label='放电'),
    Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='保持'),
]
ax5.legend(handles=legend_elements, loc='upper right', fontsize=9)

# ========== 图6: 问题总结 ==========
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')

summary_text = """
预测结果合理性评分: 7/10  [中等水平]

✗ 发现的关键问题:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 🔴 严重: 晚高峰电价异常低
   • 晚高峰(18-21时): 60.62 元/MWh
   • 谷底(0-5时): 68.18 元/MWh
   • 差值: -11.1% (应该相反!)
   • 原因: 模型特征不足或训练数据偏差

2. 🟡 中等: 晚高峰充放电决策错误
   • 第19小时(19:00-20:00)
   • 价格: 54.78 元/MWh
   • 决策: 充电 ❌ (应为放电)
   • 损失: 未能利用高价套利

3. 🟡 中等: 谷底时段未充分利用
   • 谷底(0-5时)被标记为"保持"
   • 应该充电以准备高峰放电
   • 套利效率不最优

✓ 优点:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 早高峰预测准确(88.57元/MWh)
• 充放电价差充足(66.2%)
• 数据完整性好(24小时完整)

⚠️  建议:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 不要盲目执行19:00-20:00的充电建议
2. 验证晚高峰(18-21时)的实际价格
3. 考虑手动调整晚间充放电策略
4. 使用更多特征改进模型(如负荷预测)
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('d:\\pythoncode\\预测合理性分析图表.png', dpi=300, bbox_inches='tight')
print("✓ 可视化图表已保存: 预测合理性分析图表.png")
plt.show()

# 生成数据表格
print("\n" + "="*80)
print("详细数据表格")
print("="*80)

detail_table = results[['hour_ahead', 'hour_of_day', 'predicted_price', 
                        'is_morning_peak', 'is_evening_peak', 'is_valley_hour', 
                        'storage_action']].copy()
detail_table.columns = ['小时', '时刻', '电价', '早高峰', '晚高峰', '谷底', '决策']

# 格式化
detail_table['电价'] = detail_table['电价'].apply(lambda x: f'{x:.2f}')
detail_table['早高峰'] = detail_table['早高峰'].apply(lambda x: '是' if x else '')
detail_table['晚高峰'] = detail_table['晚高峰'].apply(lambda x: '是⚠️' if x else '')  # 标记问题区间
detail_table['谷底'] = detail_table['谷底'].apply(lambda x: '是' if x else '')

print(detail_table.to_string(index=False))
