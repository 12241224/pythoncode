import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. 数据读取与预处理
# 假设数据文件名为"NSW半小时时电价预测数据管理 (1).csv"
df = pd.read_csv(r'C:\Users\admin\Desktop\NSW半小时时电价预测数据管理 (1).csv', skiprows=1)

# 打印列名以确认
print("CSV文件列名:", df.columns)

# 转换时间格式并设置为索引
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y/%m/%d %H:%M')
df.set_index('DateTime', inplace=True)

# 处理缺失值
df['WeatherType'].fillna(method='ffill', inplace=True)

# 2. 特征工程
# 添加时间特征
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['dayofweek'] = df.index.dayofweek
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# 天气类型编码
weather_dummies = pd.get_dummies(df['WeatherType'], prefix='weather')
df = pd.concat([df, weather_dummies], axis=1)

# 创建滞后特征（使用前2小时的数据）
for lag in [2, 4]:  # 1小时和2小时滞后（半小时数据间隔）
    df[f'RRP_lag{lag}'] = df['SpotPrice_RRP'].shift(lag)
    df[f'Demand_lag{lag}'] = df['TotalDemand'].shift(lag)

# 删除包含空值的行
df.dropna(inplace=True)

# 3. 准备训练数据
features = ['hour', 'minute', 'dayofweek', 'is_weekend', 'Temperature',
           'WindSpeed', 'Cloud', 'Humidity', 'Rain'] + list(weather_dummies.columns)
features += [f'RRP_lag{lag}' for lag in [2,4]] + [f'Demand_lag{lag}' for lag in [2,4]]

X = df[features]
y = df[['SpotPrice_RRP', 'TotalDemand']]

# 4. 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集（最后两天作为测试集）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=48*2, shuffle=False)  # 最后两天作为测试

# 5. 模型训练
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=200,
                         max_depth=10,
                         random_state=42,
                         n_jobs=-1)
)
model.fit(X_train, y_train)

# 6. 模型评估
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# 计算指标
mae_rrp = mean_absolute_error(y_true[:,0], y_pred[:,0])
nmae_rrp = mae_rrp / np.mean(y_true[:,0])
rmse_rrp = np.sqrt(mean_squared_error(y_true[:,0], y_pred[:,0]))
nrmse_rrp = rmse_rrp / (np.max(y_true[:,0]) - np.min(y_true[:,0]))

mae_demand = mean_absolute_error(y_true[:,1], y_pred[:,1])
nmae_demand = mae_demand / np.mean(y_true[:,1])
rmse_demand = np.sqrt(mean_squared_error(y_true[:,1], y_pred[:,1]))
nrmse_demand = rmse_demand / (np.max(y_true[:,1]) - np.min(y_true[:,1]))

r2 = r2_score(y_true, y_pred)

# 7. 预测未来数据（4月11日）
# 生成预测时间范围
future_dates = pd.date_range(start='2024-04-11 00:00',
                            end='2024-04-11 23:30',
                            freq='30T')

# 创建未来数据框架
future_df = pd.DataFrame(index=future_dates)

# 添加基本时间特征
future_df['hour'] = future_df.index.hour
future_df['minute'] = future_df.index.minute
future_df['dayofweek'] = future_df.index.dayofweek
future_df['is_weekend'] = future_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

# 初始化 'SpotPrice_RRP' 和 'TotalDemand' 列
future_df['SpotPrice_RRP'] = np.nan
future_df['TotalDemand'] = np.nan

# 使用历史数据的平均值填充其他特征（实际应用中应该使用真实预测值）
for col in ['Temperature', 'WindSpeed', 'Cloud', 'Humidity', 'Rain'] + list(weather_dummies.columns):
    future_df[col] = df[col].mean()

print("future_df 列名:", future_df.columns.tolist())

# 添加滞后特征（使用最后可用的历史数据）
last_known = df.iloc[-1]
for lag in [2, 4]:
    for i, date in enumerate(future_dates):
        if i < lag//2:  # 使用历史数据填充前几个滞后
            future_df.loc[date, f'RRP_lag{lag}'] = df['SpotPrice_RRP'].iloc[-lag + i*2]
            future_df.loc[date, f'Demand_lag{lag}'] = df['TotalDemand'].iloc[-lag + i*2]
        else:  # 使用预测值递归更新
            future_df.loc[date, f'RRP_lag{lag}'] = future_df['SpotPrice_RRP'].iloc[i - lag//2]
            future_df.loc[date, f'Demand_lag{lag}'] = future_df['TotalDemand'].iloc[i - lag//2]

# 预测未来数据
future_X = scaler_X.transform(future_df[features])
future_pred_scaled = model.predict(future_X)
future_pred = scaler_y.inverse_transform(future_pred_scaled)

# 8. 结果输出
results = pd.DataFrame({
    'DateTime': future_dates,
    'SpotPrice/RRP': future_pred[:,0],
    'TotalDemand': future_pred[:,1]
})

print("预测结果：")
print(results.to_string(index=False))

# 9. 可视化
plt.figure(figsize=(18, 12))

# 电价预测可视化
plt.subplot(2, 1, 1)
plt.plot(results['DateTime'], results['SpotPrice/RRP'], label='预测值')
plt.title('电价预测 (2024-04-11)')
plt.ylabel('电价')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.grid(True)

# 负荷预测可视化
plt.subplot(2, 1, 2)
plt.plot(results['DateTime'], results['TotalDemand'], label='预测值', color='orange')
plt.title('负荷预测 (2024-04-11)')
plt.ylabel('负荷')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.grid(True)

plt.tight_layout()
plt.show()

# 输出评估指标
print(f"""
模型评估指标：
电价预测：
MAE = {mae_rrp:.2f}
NMAE = {nmae_rrp:.2%}
RMSE = {rmse_rrp:.2f}
NRMSE = {nrmse_rrp:.2%}

负荷预测：
MAE = {mae_demand:.2f}
NMAE = {nmae_demand:.2%}
RMSE = {rmse_demand:.2f}
NRMSE = {nrmse_demand:.2%}

R² Score = {r2:.2f}
""")