import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import lightgbm as lgb
def calculate_metrics(y_true, y_pred):
    """
    计算多个评估指标
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    r2 = r2_score(y_true, y_pred)
    nmae = mae / (np.max(y_true) - np.min(y_true))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))
    imape_values = []
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val < 0:
            if pred_val < 0:
                imape_values.append(0)
            elif 0 <= pred_val <= 30:
                imape_values.append(0.2)
            else:
                imape_values.append(1)
        elif 0 <= true_val <= 30:
            if pred_val < 0:
                imape_values.append(0.2)
            elif 0 <= pred_val <= 30:
                imape_values.append(0)
            else:
                imape_values.append(min(np.abs((30 - pred_val) / 30), 1))
        else:
            imape_values.append(min(np.abs((true_val - pred_val) / true_val), 1))
    imape = np.mean(imape_values)
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'NMAE': nmae,
        'NRMSE': nrmse,
        'IMAPE': imape
    }
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 1. 数据加载
data = pd.read_csv(
    r"C:\Users\admin\Desktop\实际数据.csv",
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

# 评估指标（合并自 metrics(1).py）
metrics = calculate_metrics(y_test.values, y_pred)
print("详细评估指标：")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# 保存评估指标到 Excel
import pandas as pd
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['值'])
metrics_df.index.name = '指标'
metrics_df.to_excel('evaluation_metrics.xlsx')
print('评估指标已保存到 evaluation_metrics.xlsx')

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


# 在原有代码末尾添加保存代码
import dill

# 保存关键组件
model_components = {
    'model': model,
    'label_encoder': LabelEncoder().fit(data['WeatherType']),
    'features': features,
    'target': target
}

# 保存到dill文件
with open('lightgbm_model_components.dill', 'wb') as f:
    dill.dump(model_components, f)
print("LightGBM模型组件已保存")

# 加载示例
with open('lightgbm_model_components.dill', 'rb') as f:
    components = dill.load(f)

# 使用加载的组件进行预测
def predict_with_saved_model(new_data, components):
    """使用保存的组件进行预测"""
    # 复制数据并处理
    df = new_data.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.sort_values('DateTime')
    
    # 时间特征
    df['hour'] = df['DateTime'].dt.hour
    df['dayofweek'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['day'] = df['DateTime'].dt.day
    
    # 编码
    df['WeatherType'] = components['label_encoder'].transform(df['WeatherType'])
    
    # 滞后特征
    df['SpotPrice_RRP_lag1'] = df['SpotPrice_RRP'].shift(1)
    df['TotalDemand_lag1'] = df['TotalDemand'].shift(1)
    
    # 数值转换
    for col in ['TotalDemand', 'Temperature', 'WindSpeed', 'Cloud', 'Humidity', 'Rain', 
                'SpotPrice_RRP', 'SpotPrice_RRP_lag1', 'TotalDemand_lag1']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    X = df[components['features']]
    
    return components['model'].predict(X)

# 使用保存的模型预测
predictions = predict_with_saved_model(data, components)