import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\admin\Desktop\NSW半小时时电价预测数据管理 (1).csv', parse_dates=['DateTime'])

# 处理缺失值
df['WeatherType'].fillna('晴', inplace=True)  # 天气类型空值填充为"晴"
df = df.sort_values('DateTime')  # 确保时间顺序

# 提取时间特征
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek  # 周一=0, 周日=6
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 天气类型编码
encoder = OneHotEncoder(sparse=False)
weather_encoded = encoder.fit_transform(df[['WeatherType']])
weather_cols = encoder.get_feature_names_out(['WeatherType'])
df_weather = pd.DataFrame(weather_encoded, columns=weather_cols, index=df.index)
df = pd.concat([df, df_weather], axis=1)

# 选择特征列
features = ['hour', 'day_of_week', 'is_weekend', 'Temperature', 
           'WindSpeed', 'Cloud', 'Humidity', 'Rain'] + list(weather_cols)

# 标准化数值特征
scaler = StandardScaler()
df[['Temperature', 'WindSpeed', 'Cloud', 'Humidity', 'Rain']] = scaler.fit_transform(
    df[['Temperature', 'WindSpeed', 'Cloud', 'Humidity', 'Rain']])

# 电价和负荷的滞后特征
for lag in [1, 24, 48]:  # 1小时前、24小时前、48小时前
    df[f'price_lag_{lag}'] = df['SpotPrice/RRP'].shift(lag)
    df[f'load_lag_{lag}'] = df['TotalDemand'].shift(lag)

# 删除包含NaN的行（首两天的数据因滞后被舍弃）
df = df.dropna()

# 最后一天作为测试集（模拟第8天预测）
train = df[df['DateTime'] < '2024-04-07']
test = df[df['DateTime'] >= '2024-04-07']

X_train = train[features + ['price_lag_1', 'price_lag_24', 'price_lag_48']]
y_price_train = train['SpotPrice/RRP']
y_load_train = train['TotalDemand']

X_test = test[features + ['price_lag_1', 'price_lag_24', 'price_lag_48']]

# 电价预测模型
model_price = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_price.fit(X_train, y_price_train)

# 负荷预测模型
model_load = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_load.fit(X_train, y_load_train)

# 生成第8天的时间序列（2024-04-08 00:00:00 ~ 23:30:00）
future_dates = pd.date_range(start='2024-04-08 00:00:00', periods=48, freq='30min')
future_data = pd.DataFrame({'DateTime': future_dates})

# 复制最后一天的天气数据（假设天气相似）
last_day_weather = df[df['DateTime'].dt.date == pd.to_datetime('2024-04-07').date()].iloc[:48]
future_data = pd.concat([future_data, last_day_weather.drop('DateTime', axis=1)], axis=1)

# 添加滞后特征（使用预测值动态更新）
for i in range(len(future_data)):
    if i == 0:
        future_data.loc[i, 'price_lag_1'] = df['SpotPrice/RRP'].iloc[-1]
        future_data.loc[i, 'price_lag_24'] = df['SpotPrice/RRP'].iloc[-48]
        future_data.loc[i, 'price_lag_48'] = df['SpotPrice/RRP'].iloc[-96]
    else:
        future_data.loc[i, 'price_lag_1'] = future_data.loc[i-1, 'predicted_price']
        future_data.loc[i, 'price_lag_24'] = future_data.loc[max(i-48,0), 'predicted_price']
        future_data.loc[i, 'price_lag_48'] = future_data.loc[max(i-96,0), 'predicted_price']

# 预测
future_data['predicted_price'] = model_price.predict(future_data[features + ['price_lag_1', 'price_lag_24', 'price_lag_48']])
future_data['predicted_load'] = model_load.predict(future_data[features + ['price_lag_1', 'price_lag_24', 'price_lag_48']])

plt.figure(figsize=(14, 6))
# 电价预测
plt.subplot(1,2,1)
plt.plot(future_data['DateTime'], future_data['predicted_price'], label='Predicted Price')
plt.title('Day 8 Electricity Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price ($/MWh)')

# 负荷预测
plt.subplot(1,2,2)
plt.plot(future_data['DateTime'], future_data['predicted_load'], label='Predicted Load')
plt.title('Day 8 Electricity Load Prediction')
plt.xlabel('Time')
plt.ylabel('Load (MWh)')

plt.tight_layout()
plt.show()