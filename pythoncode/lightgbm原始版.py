import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 1. 数据加载
data = pd.read_csv(
    r'd:\pythoncode\pythoncode\NSW半小时时电价预测数据管理 (1).csv',
    skiprows=0,
    names=['DateTime','SpotPrice_RRP','TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain']
)
    # 自动将相关列转换为数值类型，避免 LightGBM 报错
for col in ['TotalDemand', 'Temperature', 'WindSpeed', 'Cloud', 'Humidity', 'Rain', 'SpotPrice_RRP', 'SpotPrice_RRP_lag1', 'TotalDemand_lag1']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# 时间戳处理
data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
data = data.sort_values('DateTime')

# 构造时间特征
data['hour'] = data['DateTime'].dt.hour
data['dayofweek'] = data['DateTime'].dt.dayofweek
data['month'] = data['DateTime'].dt.month
data['day'] = data['DateTime'].dt.day

# 编码类别特征
data['WeatherType'] = LabelEncoder().fit_transform(data['WeatherType'])

# 构造滞后特征（如前1步电价、负荷）

# 构造滞后特征（如前1步电价、负荷）
data['SpotPrice_RRP_lag1'] = data['SpotPrice_RRP'].shift(1)
data['TotalDemand_lag1'] = data['TotalDemand'].shift(1)
# 保证滞后特征为数值类型
data['SpotPrice_RRP_lag1'] = pd.to_numeric(data['SpotPrice_RRP_lag1'], errors='coerce')
data['TotalDemand_lag1'] = pd.to_numeric(data['TotalDemand_lag1'], errors='coerce')

# 去除首行缺失
data = data.dropna()

# 2. 特征与标签
target = 'SpotPrice_RRP'
features = ['TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain',
            'hour','dayofweek','month','day','SpotPrice_RRP_lag1','TotalDemand_lag1']

X = data[features]
y = data[target]

# 3. 划分训练/测试集（最后一天为测试集）
train_size = int(len(data)*0.95)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 4. LightGBM模型训练
dataset_train = lgb.Dataset(X_train, y_train)
dataset_test = lgb.Dataset(X_test, y_test, reference=dataset_train)
params = {
    'objective': 'regression',
    'metric': 'mae',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}
model = lgb.LGBMRegressor(
    objective='regression',
    boosting_type='gbdt',
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_estimators=200
)
# 训练，支持早停
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='mae'
)

# 5. 预测与评估
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print(f'MAE: {mae:.2f}, RMSE: {rmse:.2f}')

# 6. 保存结果
data_test = data.iloc[train_size:].copy()
data_test['PredictedPrice'] = y_pred
data_test[['DateTime','SpotPrice_RRP','PredictedPrice']].to_csv('lightgbm_predicted_prices.csv', index=False)
print('预测结果已保存到 lightgbm_predicted_prices.csv')

# 7. 可选：分位数预测（不确定性区间）
params_quantile = params.copy()
params_quantile['objective'] = 'quantile'
params_quantile['alpha'] = 0.9  # 预测90%分位数
model_q = lgb.train(params_quantile, dataset_train, num_boost_round=200)
y_q_pred = model_q.predict(X_test)
data_test['PredictedPrice_90q'] = y_q_pred
data_test[['DateTime','SpotPrice_RRP','PredictedPrice_90q']].to_csv('lightgbm_predicted_prices_90q.csv', index=False)

print('分位数预测结果已保存到 lightgbm_predicted_prices_90q.csv')

# 8. 绘制预测与实际对比图
import matplotlib.pyplot as plt
# 8. 绘制预测与实际对比图，设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
plt.figure(figsize=(12, 6))
plt.plot(data_test['DateTime'], data_test['SpotPrice_RRP'], label='实际价格', color='blue')
plt.plot(data_test['DateTime'], data_test['PredictedPrice'], label='预测价格', color='red', linestyle='--')
plt.xlabel('时间')
plt.ylabel('电价')
plt.title('LightGBM电价预测 vs 实际')
plt.legend()
plt.tight_layout()
plt.savefig('lightgbm_price_comparison.png')
plt.show()