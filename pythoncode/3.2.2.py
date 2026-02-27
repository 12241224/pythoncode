import cvxpy as cp
import numpy as np

# ================== 参数定义 ==================
T = 24  # 时间周期（24小时）
N_G = 3  # 火电机组数量
G_units = list(range(N_G))  # 机组集合 [0, 1, 2]

# 机组参数 (示例值)
P_min = np.array([20, 30, 40])       # 最小出力 (MW)
P_max = np.array([100, 150, 200])    # 最大出力 (MW)
RU = np.array([40, 50, 60])          # 最大上爬坡速率 (MW/h)
RD = np.array([30, 40, 50])          # 最大下爬坡速率 (MW/h)
SU_cost = np.array([500, 800, 1000]) # 启动成本 ($)
SD_cost = np.array([300, 500, 700])  # 停机成本 ($)
Ton_min = np.array([4, 5, 6])        # 最短开机时间 (h)
Toff_min = np.array([3, 4, 5])       # 最短停机时间 (h)

# 发电成本线性化参数（方案2）
k = np.array([0.5, 0.6, 0.7])  # 斜率 ($/MW)
h = np.array([100, 150, 200])  # 截距 ($)

# 负荷参数
P_load = np.random.randint(200, 400, T)  # 每小时负荷 (MW)
alpha_RS = 0.1  # 备用系数

# ================== 变量定义 ==================
# 机组状态变量 (0-1)
x = cp.Variable((N_G, T), boolean=True)
# 机组出力
P = cp.Variable((N_G, T), nonneg=True)
# 启动/停机变量 (方案1)
u = cp.Variable((N_G, T), boolean=True)
d = cp.Variable((N_G, T), boolean=True)
# 松弛变量 (切负荷、失备用)
P_cut = cp.Variable(T, nonneg=True)  # 切负荷功率
P_RS_loss = cp.Variable(T, nonneg=True)  # 失备用功率

# ================== 目标函数 ==================
# 发电成本（线性化方案2）
cost_gen = cp.sum([k[i] * P[i, t] + h[i] * x[i, t] for i in G_units for t in range(T)])
# 启停成本（方案1）
cost_SU_SD = cp.sum([SU_cost[i] * u[i, t] + SD_cost[i] * d[i, t] for i in G_units for t in range(T)])

# 松弛惩罚项（权重需足够大）
penalty_cut = 1e5 * cp.sum(P_cut)     # 切负荷惩罚
penalty_RS = 1e4 * cp.sum(P_RS_loss)  # 失备用惩罚

total_cost = cost_gen + cost_SU_SD + penalty_cut + penalty_RS
objective = cp.Minimize(total_cost)

# ================== 约束条件 ==================
constraints = []

# --------- 机组状态逻辑约束（方案1）---------
for i in G_units:
    for t in range(1, T):
        constraints += [x[i,t] - x[i,t-1] == u[i,t] - d[i,t]]
    # 初始状态假设（可根据实际情况修改）
    constraints += [x[i,0] == 0]  # 初始关机

# --------- 出力约束 ---------
for i in G_units:
    for t in range(T):
        constraints += [P[i,t] >= P_min[i] * x[i,t]]
        constraints += [P[i,t] <= P_max[i] * x[i,t]]

# --------- 爬坡约束（方案1）---------
for i in G_units:
    for t in range(1, T):
        constraints += [P[i,t] - P[i,t-1] <= RU[i] * x[i,t-1] + P_max[i] * u[i,t]]
        constraints += [P[i,t-1] - P[i,t] <= RD[i] * x[i,t] + P_max[i] * d[i,t]]

# --------- 最短开停机时间（方案1）---------
for i in G_units:
    for t in range(T):
        # 开机时间约束
        if t + Ton_min[i] <= T:
            constraints += [cp.sum(x[i, t:t+Ton_min[i]]) >= Ton_min[i] * u[i,t]]
        # 停机时间约束
        if t + Toff_min[i] <= T:
            constraints += [cp.sum(1 - x[i, t:t+Toff_min[i]]) >= Toff_min[i] * d[i,t]]

# --------- 功率平衡（含松弛）---------
for t in range(T):
    constraints += [cp.sum(P[:, t]) + P_cut[t] == P_load[t]]

# --------- 备用约束（含松弛）---------
for t in range(T):
    constraints += [cp.sum((P_max[i] - P[i,t]) * x[i,t] for i in G_units) + P_RS_loss[t] >= alpha_RS * P_load[t]]

# ================== 求解与结果 ==================
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CBC, verbose=True)  # 推荐CBC求解混合整数问题

# ------ 输出结果 ------
if problem.status == "optimal":
    print("总成本: $", problem.value)
    print("机组出力计划:")
    print(P.value)
    print("启停状态:")
    print(x.value)
else:
    print("求解失败，状态:", problem.status)