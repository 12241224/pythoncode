import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 模拟增强后的新能源出力数据
np.random.seed(42)
hours = 24

# 模拟各类电源出力（模拟增强后的效果）
thermal_gen = np.array([6500, 6200, 5800, 5500, 5400, 5600, 6200, 7000, 7500, 7800, 8000, 8200, 
                       8300, 8400, 8200, 8000, 7800, 7600, 7400, 7200, 7000, 6800, 6600, 6500])

# 大幅增强的风电出力
wind_gen = np.array([200, 180, 220, 250, 280, 300, 150, 100, 120, 150, 180, 200,
                    220, 250, 280, 300, 320, 350, 300, 280, 250, 230, 210, 200])

# 大幅增强的光伏出力（白天高峰）
solar_gen = np.array([0, 0, 0, 0, 0, 0, 50, 150, 300, 450, 500, 480,
                     460, 420, 380, 300, 200, 80, 20, 0, 0, 0, 0, 0])

# 水电出力
hydro_gen = np.array([400, 400, 400, 400, 400, 400, 600, 800, 800, 600, 400, 400,
                     400, 400, 600, 800, 800, 600, 400, 400, 400, 400, 400, 400])

# 储能放电
ess_discharge = np.array([0, 0, 0, 0, 0, 100, 200, 300, 200, 100, 0, 0,
                         0, 0, 100, 200, 300, 200, 100, 0, 0, 0, 0, 0])

# 储能充电
ess_charge = np.array([200, 150, 100, 50, 0, 0, 0, 0, 0, 0, 50, 100,
                      150, 200, 0, 0, 0, 0, 0, 100, 150, 200, 200, 200])

# 总负荷
total_load = np.array([5371, 5050, 4810, 4730, 4730, 4810, 5932, 6894, 7616, 7696, 7696, 7616,
                      7616, 7616, 7455, 7536, 7936, 8017, 8017, 7696, 7295, 6654, 5852, 5050])

# 创建堆积柱状图
fig, ax = plt.subplots(figsize=(15, 8))

# 时间轴
hours_range = np.arange(hours)
width = 0.8

# 绘制堆积柱状图
bottom = np.zeros(hours)

# 火电（红色）
p1 = ax.bar(hours_range, thermal_gen, width, bottom=bottom, label='Power_ThermalGen', 
           color='#ff4444', alpha=0.8)
bottom += thermal_gen

# 水电（蓝色）
p2 = ax.bar(hours_range, hydro_gen, width, bottom=bottom, label='Power_RHD', 
           color='#4444ff', alpha=0.8)
bottom += hydro_gen

# 风电（绿色）
p3 = ax.bar(hours_range, wind_gen, width, bottom=bottom, label='Power_WT', 
           color='#44ff44', alpha=0.8)
bottom += wind_gen

# 光伏（黄色）
p4 = ax.bar(hours_range, solar_gen, width, bottom=bottom, label='Power_PV', 
           color='#ffff44', alpha=0.8)
bottom += solar_gen

# 储能放电（灰色）
p5 = ax.bar(hours_range, ess_discharge, width, bottom=bottom, label='Power_StoreDis', 
           color='#888888', alpha=0.8)
bottom += ess_discharge

# 储能充电（负值，浅灰色）
p6 = ax.bar(hours_range, -ess_charge, width, label='Power_StoreCharge', 
           color='#cccccc', alpha=0.6)

# 添加总负荷曲线（黑色虚线）
ax.plot(hours_range, total_load, 'k--', linewidth=2, label='Total Demand')

# 计算备用容量
total_gen = thermal_gen + wind_gen + solar_gen + hydro_gen + ess_discharge
available_reserve = total_gen - total_load

# 添加备用容量要求线（绿色虚线）
reserve_req = total_load * 0.1  # 10%备用要求
ax.plot(hours_range, total_load + reserve_req, 'g:', linewidth=2, 
       label='Reserve Requirement', alpha=0.7)

# 填充可用备用区域
ax.fill_between(hours_range, total_load + reserve_req, total_gen, 
               alpha=0.2, color='lightgreen', label='Available Reserve')

# 设置图表属性
ax.set_title('Generation Schedule and Reserve Margin (Enhanced Renewable)', fontsize=14, fontweight='bold')
ax.set_xlabel('Hour', fontsize=12)
ax.set_ylabel('Power (MW)', fontsize=12)

# 添加统计信息
max_demand = max(total_load)
min_reserve = min(available_reserve)
wind_total = sum(wind_gen)
solar_total = sum(solar_gen)
ax.text(0.02, 0.98, f'Max Demand: {max_demand:.1f} MW\nMin Reserve: {min_reserve:.1f} MW\nWind Gen: {wind_total:.0f} MWh\nSolar Gen: {solar_total:.0f} MWh', 
       transform=ax.transAxes, fontsize=10, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 设置图例
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)

# 设置网格
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlim(-0.5, hours-0.5)
ax.set_xticks(range(0, hours, 2))

# 紧凑布局
plt.tight_layout()

# 保存图表
plt.savefig('Enhanced_Generation_Mix_Preview.png', dpi=300, bbox_inches='tight')
print("增强型发电组合预览图已保存为 Enhanced_Generation_Mix_Preview.png")
print(f"风电总发电量: {wind_total:.0f} MWh")
print(f"光伏总发电量: {solar_total:.0f} MWh")
print(f"水电总发电量: {sum(hydro_gen):.0f} MWh")
plt.show()
