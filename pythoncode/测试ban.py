import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.ticker import MultipleLocator

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 实验八数据 ====================
# 8-1 负载实验数据 (工作特性)
data_load = np.array([
    [5.12, 4.89, 5.02, 765.3, 1921, 1.55, 1437, 5.01, 2686.3, 2287.5, 0.8515, 0.814, 0.042],
    [4.52, 4.3, 4.29, 590.7, 1625, 1.3, 1450, 4.37, 2215.7, 1935.9, 0.8737, 0.77, 0.033],
    [4.08, 3.88, 3.85, 418.6, 1434, 1.1, 1460, 3.94, 1852.6, 1649.4, 0.8903, 0.72, 0.0267],
    [3.67, 3.5, 3.61, 235.6, 1310, 0.9, 1466, 3.59, 1563.6, 1355, 0.8666, 0.66, 0.0267],
    [3.4, 3.24, 3.22, 97.76, 1135, 0.7, 1474, 3.29, 1232.76, 1059.7, 0.86, 0.57, 0.017],
    [3.11, 3.01, 2.91, -81.2, 938.2, 0.5, 1482, 3.01, 857, 761, 0.888, 0.43, 0.012],
    [2.88, 2.85, 2.75, -337, 697.4, 0.15, 1495, 2.83, 360.4, 230.3, 0.639, 0.19, 0.0033]
])

# 8-2 空载实验数据
data_no_load = np.array([
    [380.2, 381.4, 381, 2.88, 2.86, 2.7, -418, 609.5, 380.8666667, 2.813333333, 191.5, 0.103187415],
    [339.9, 341.5, 342.8, 2.38, 2.22, 2.21, -329, 470.9, 341.4, 2.27, 141.9, 0.105717074],
    [302.5, 303.8, 304.3, 1.97, 1.85, 1.86, -241, 351.9, 303.5333333, 1.893333333, 110.9, 0.111416673],
    [261.3, 263.7, 264.3, 1.63, 1.46, 1.6, -182, 273.1, 263.1, 1.563333333, 91.1, 0.127878653],
    [222.5, 224.8, 226, 1.3, 1.14, 1.4, -137, 203.6, 224.4333333, 1.28, 66.6, 0.133853278],
    [178.8, 183, 183.8, 1.03, 0.85, 1.18, -91.6, 155.1, 181.8666667, 1.02, 63.5, 0.19763896],
    [142.6, 145.4, 145.5, 0.79, 0.71, 0.92, -45.5, 98.99, 144.5, 0.806666667, 53.49, 0.264949285],
    [81.15, 84.27, 86.28, 0.55, 0.42, 0.72, -11.4, 56.77, 83.9, 0.563333333, 45.37, 0.554234404],
    [95.13, 99.18, 100.2, 0.57, 0.48, 0.77, -19.5, 66.58, 98.17, 0.606666667, 47.08, 0.45641472]
])

# 8-3 堵转实验数据
data_block = np.array([
    [77.16, 78.77, 78.48, 4.97, 5.09, 5.12, 29.14, 366.3, 78.13666667, 5.06, 395.44, 0.577467188],
    [69.33, 70.62, 70.99, 4.45, 4.54, 4.57, 26.1, 291.6, 70.31333333, 4.52, 317.7, 0.577155987],
    [60.01, 60.45, 60.08, 3.79, 3.91, 3.87, 18.72, 212.3, 60.18, 3.856666667, 231.02, 0.574694981],
    [53.13, 53.44, 53.88, 3.38, 3.43, 3.43, 15.16, 166.2, 53.48333333, 3.413333333, 181.36, 0.573583237],
    [47.3, 48.24, 47.71, 2.97, 3.07, 3.07, 9.75, 133.8, 47.75, 3.036666667, 143.55, 0.571590248],
    [38.74, 39.15, 39.56, 2.44, 2.49, 2.5, 6.24, 88.15, 39.15, 2.476666667, 94.39, 0.562054945],
    [31.02, 30.97, 30.7, 1.92, 2, 1.694, 3.96, 34, 30.89666667, 1.871333333, 37.96, 0.379066613],
    [16.6, 17.26, 17.64, 1.03, 1.03, 1.11, 0.13, 16.27, 17.16666667, 1.056666667, 16.4, 0.522001774]
])

# ==================== 第四问：参数计算 ====================
print("========== 第四问：等效电路参数计算 ==========")

# 1. 从空载实验中求出Rm、Xm
print("\n1. 空载实验参数计算：")
U0 = data_no_load[:, 8]      # 空载电压 (V)
I0 = data_no_load[:, 9]      # 空载电流 (A)
P0 = data_no_load[:, 10]     # 空载功率 (W)

# 计算U0²
U0_squared = U0**2

# 计算p0' = P0 - 3I0²R1 (假设R1已知，这里需要根据实际测量值修改)
# 注意：这里需要实际的定子电阻R1值，假设为2.0Ω（示例值）
R1 = 3.02  # Ω，定子每相电阻（实际应根据实验测量值）
p0_prime = P0 - 3 * I0**2 * R1

# 绘制p0' = f(U0²)曲线
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(U0_squared/1000, p0_prime, 'bo-', linewidth=2, markersize=8, label='实验数据')
ax4.set_xlabel('U₀²/1000 (V²/k)')
ax4.set_ylabel("p₀' (W)")
ax4.set_title("图8-4 p₀' = f(U₀²) 曲线")
ax4.grid(True, alpha=0.3)
ax4.legend()

# 添加数据点标签
for i, (x, y) in enumerate(zip(U0_squared/1000, p0_prime)):
    ax4.text(x, y, f'{i+1}', fontsize=9, ha='right', va='bottom')

# 线性拟合外推求机械损耗p_Ω
# 选择低压段数据进行线性拟合（通常U0² < 0.2U_N²）
low_voltage_idx = U0_squared < 0.2 * (380**2)  # 380V为额定电压
if np.sum(low_voltage_idx) >= 3:  # 至少有3个点
    coeff = np.polyfit(U0_squared[low_voltage_idx], p0_prime[low_voltage_idx], 1)
    p_fit = np.poly1d(coeff)
    
    # 绘制拟合直线
    x_fit = np.linspace(0, max(U0_squared), 100)
    y_fit = p_fit(x_fit)
    ax4.plot(x_fit/1000, y_fit, 'r--', linewidth=1.5, label=f'拟合直线: y={coeff[0]:.4f}x+{coeff[1]:.2f}')
    
    # 外推到U0²=0得到机械损耗p_Ω
    p_omega = coeff[1]  # 截距即为机械损耗
    print(f"  机械损耗 p_Ω = {p_omega:.2f} W")
    
    # 在额定电压UN=380V时的铁耗
    U_N_squared = 380**2
    p_Fe_at_UN = coeff[0] * U_N_squared
    print(f"  额定电压时铁耗 p_Fe = {p_Fe_at_UN:.2f} W")
    
    # 在额定电压UN=380V时的p0'
    p0_prime_at_UN = p_fit(U_N_squared)
    
    # 计算Rm
    # 首先需要额定电压下的空载电流I0
    # 插值得到U0=380V时的I0
    I0_interp = interp1d(U0, I0, kind='linear', fill_value='extrapolate')
    I0_at_UN = I0_interp(380)
    print(f"  额定电压时空载电流 I0 = {I0_at_UN:.3f} A")
    
    # 计算Rm (励磁电阻)
    Rm = p_Fe_at_UN / (3 * I0_at_UN**2)
    print(f"  励磁电阻 Rm = {Rm:.2f} Ω")
else:
    print("  警告：低压数据点不足，无法进行线性拟合")

# 2. 从堵转实验中求出R1, X1σ, R2', X2σ'等参数
print("\n2. 堵转实验参数计算：")
UK = data_block[:, 8]      # 堵转电压 (V)
IK = data_block[:, 9]      # 堵转电流 (A)
PK = data_block[:, 10]     # 堵转功率 (W)

# 插值得到IK = IN时的UK、PK
# 假设额定电流IN = 5.01A (从负载实验第一点获得)
I_N = 5.01  # A

# 插值函数
UK_interp = interp1d(IK, UK, kind='linear', fill_value='extrapolate')
PK_interp = interp1d(IK, PK, kind='linear', fill_value='extrapolate')

UK_at_IN = UK_interp(I_N)
PK_at_IN = PK_interp(I_N)

print(f"  额定电流 I_N = {I_N} A 时的参数：")
print(f"  堵转电压 U_K = {UK_at_IN:.2f} V")
print(f"  堵转功率 P_K = {PK_at_IN:.2f} W")

# 计算堵转阻抗参数
# 对于星形接法，相电压 = 线电压/√3，相电流 = 线电流
U_ph = UK_at_IN / np.sqrt(3)  # 相电压
I_ph = I_N  # 相电流
Z_K = U_ph / I_ph  # 堵转阻抗
R_K = PK_at_IN / (3 * I_ph**2)  # 堵转电阻
X_K = np.sqrt(Z_K**2 - R_K**2)  # 堵转电抗

print(f"  堵转阻抗 Z_K = {Z_K:.4f} Ω")
print(f"  堵转电阻 R_K = {R_K:.4f} Ω")
print(f"  堵转电抗 X_K = {X_K:.4f} Ω")

# 假设定转子漏抗相等：X1σ = X2σ' = X_K/2
X1_sigma = X_K / 2
X2_sigma_prime = X_K / 2

# 归算到定子边的转子电阻 R2' = R_K - R1
# 注意：这里R1应该使用测量值，我们使用上面假设的R1=2.0Ω
R2_prime = R_K - R1

print(f"  定子漏抗 X1σ = {X1_sigma:.4f} Ω")
print(f"  转子漏抗(归算) X2σ' = {X2_sigma_prime:.4f} Ω")
print(f"  转子电阻(归算) R2' = {R2_prime:.4f} Ω")

# 3. 计算励磁电抗Xm
print("\n3. 励磁电抗Xm计算：")
# 使用额定电压下的空载参数计算Xm
# 空载阻抗 Z0 = U0/(√3 * I0)
U0_at_UN = 380  # V
I0_at_UN = I0_at_UN  # A (上面已计算)
Z0 = U0_at_UN / (np.sqrt(3) * I0_at_UN)  # 空载阻抗

# 空载电阻 R0 = P0/(3 * I0²)
# 需要额定电压下的空载功率P0
P0_interp = interp1d(U0, P0, kind='linear', fill_value='extrapolate')
P0_at_UN = P0_interp(380)
R0 = P0_at_UN / (3 * I0_at_UN**2)

# 空载电抗 X0 = √(Z0² - R0²)
X0 = np.sqrt(Z0**2 - R0**2)

# 励磁电抗 Xm = X0 - X1σ
Xm = X0 - X1_sigma

print(f"  额定电压时空载功率 P0 = {P0_at_UN:.2f} W")
print(f"  空载阻抗 Z0 = {Z0:.4f} Ω")
print(f"  空载电阻 R0 = {R0:.4f} Ω")
print(f"  空载电抗 X0 = {X0:.4f} Ω")
print(f"  励磁电抗 Xm = {Xm:.4f} Ω")

# 4. 绘制T型等效电路图
print("\n4. T型等效电路图参数汇总：")
print("  " + "="*50)
print("  R1 = {:.4f} Ω (定子电阻，假设值)".format(R1))
print("  X1σ = {:.4f} Ω (定子漏抗)".format(X1_sigma))
print("  Rm = {:.4f} Ω (励磁电阻)".format(Rm))
print("  Xm = {:.4f} Ω (励磁电抗)".format(Xm))
print("  R2' = {:.4f} Ω (转子电阻归算值)".format(R2_prime))
print("  X2σ' = {:.4f} Ω (转子漏抗归算值)".format(X2_sigma_prime))
print("  " + "="*50)

# 绘制等效电路示意图
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.axis('off')
ax5.set_title("三相异步电动机T型等效电路图", fontsize=14, fontweight='bold')

# 绘制电路图
# 设置坐标
x_vals = [0, 2, 4, 6, 8]
y_vals = [0, -2, 0, 2, 0]

# 绘制电阻、电抗符号
ax5.text(1, 0.5, f"R1={R1:.2f}Ω", fontsize=10, ha='center', va='center', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
ax5.text(1, -0.5, f"X1σ={X1_sigma:.2f}Ω", fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

ax5.text(3, 0.5, f"Rm={Rm:.2f}Ω", fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
ax5.text(3, -0.5, f"Xm={Xm:.2f}Ω", fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

ax5.text(5, 0.5, f"R2'={R2_prime:.2f}Ω", fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
ax5.text(5, -0.5, f"X2σ'={X2_sigma_prime:.2f}Ω", fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

# 绘制连线
ax5.plot([0, 8], [0, 0], 'k-', linewidth=2)  # 主水平线
ax5.plot([2, 2], [-1, 1], 'k-', linewidth=1)  # 垂直线1
ax5.plot([4, 4], [-1, 1], 'k-', linewidth=1)  # 垂直线2
ax5.plot([6, 6], [-1, 1], 'k-', linewidth=1)  # 垂直线3

# 添加输入输出标记
ax5.text(-0.5, 0, "U₁\nI₁", fontsize=12, ha='center', va='center', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
ax5.text(8.5, 0, "E₂'\n(1-s)/s·R₂'", fontsize=10, ha='center', va='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

# 添加注释
ax5.text(4, -3, "注：图中参数为计算值，实际应以实验测量为准", 
         fontsize=9, ha='center', va='center', style='italic')

plt.tight_layout()

# ==================== 显示所有图表 ====================
print("\n正在生成图表...")
plt.show()

# ==================== 保存图表到文件 ====================
fig4.savefig('p0_prime_vs_U0_squared.png', dpi=300, bbox_inches='tight')
fig5.savefig('T型等效电路图.png', dpi=300, bbox_inches='tight')

print("\n所有图表已生成并保存为PNG文件:")
print("1. p0_prime_vs_U0_squared.png")
print("2. T型等效电路图.png")
print("\n注意：计算中使用了假设的定子电阻R1=2.0Ω，实际计算时应使用测量值。")
print("请根据实验测量的定子电阻值修改R1变量，重新运行计算。")