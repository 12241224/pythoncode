import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

def calculate_metrics(y_true, y_pred, feature_name):
    """
    增强版评估指标计算函数（支持多特征评估）
    :param y_true: 真实值数组
    :param y_pred: 预测值数组
    :param feature_name: 特征名称（用于输出）
    :return: 包含多个评估指标的字典
    """
    metrics = {}
    
    # 基础指标
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # 标准化指标
    value_range = np.max(y_true) - np.min(y_true)
    metrics['NMAE'] = metrics['MAE'] / value_range
    metrics['NRMSE'] = metrics['RMSE'] / value_range
    
    # MAPE（处理零值情况）
    epsilon = 1e-10  # 防止除以零
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    metrics['MAPE(%)'] = mape
    
    # IMAPE（根据特征类型调整计算逻辑）
    imape_values = []
    for true_val, pred_val in zip(y_true, y_pred):
        if "Price" in feature_name:
            # 电价特殊处理（可能包含负值）
            if true_val < 0:
                if pred_val < 0:
                    imape_values.append(0)
                elif 0 <= pred_val <= 30:
                    imape_values.append(0.2)
                else:
                    imape_values.append(1)
            else:
                imape_values.append(min(np.abs((true_val - pred_val) / (true_val + epsilon)), 1))
        else:
            # 负荷处理（通常为正值）
            imape_values.append(min(np.abs((true_val - pred_val) / (true_val + epsilon)), 1))
    
    metrics['IMAPE'] = np.mean(imape_values)
    
    return metrics

# 1. 数据读取与预处理
df = pd.read_csv(r'C:\Users\admin\Desktop\NSW半小时时电价预测数据管理 (1).csv', skiprows=1)

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

# 创建滞后特征
for lag in [2, 4]:
    df[f'RRP_lag{lag}'] = df['SpotPrice_RRP'].shift(lag)
    df[f'Demand_lag{lag}'] = df['TotalDemand'].shift(lag)

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

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=48*2, shuffle=False)

# 5. 模型训练
model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=200, 
                         max_depth=10,
                         random_state=42,
                         n_jobs=-1)
)
model.fit(X_train, y_train)

# 6. 模型评估和结果保存
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

# 计算各特征指标
metrics_results = {}
for idx, feature in enumerate(['SpotPrice/RRP', 'TotalDemand']):
    metrics_results[feature] = calculate_metrics(
        y_true[:, idx], 
        y_pred[:, idx],
        feature
    )

# 保存评估结果到Excel
with pd.ExcelWriter('预测结果_评估指标.xlsx') as writer:
    # 保存预测结果
    future_dates = pd.date_range(start='2024-04-11 00:00', end='2024-04-11 23:30', freq='30T')
    future_df = pd.DataFrame(index=future_dates)
    
# 创建未来数据框架
    future_df = pd.DataFrame(index=future_dates)

    # 添加基本时间特征
    future_df['hour'] = future_df.index.hour
    future_df['minute'] = future_df.index.minute
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['is_weekend'] = future_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # 初始化目标列
    future_df['SpotPrice_RRP'] = np.nan
    future_df['TotalDemand'] = np.nan

    # 填充数值型特征（使用历史数据最后一周的平均值）
    historical_mean = df.last('7D').mean()
    for col in ['Temperature', 'WindSpeed', 'Cloud', 'Humidity', 'Rain']:
        future_df[col] = historical_mean[col]

    # 填充天气类型（使用最后已知天气模式）
    last_weather = df['WeatherType'].mode()[0]
    weather_dummy_cols = [f'weather_{w}' for w in df['WeatherType'].unique() if pd.notna(w)]
    future_df[weather_dummy_cols] = 0
    future_df[f'weather_{last_weather}'] = 1

    # 动态生成滞后特征
    last_4_steps = df[['SpotPrice_RRP', 'TotalDemand']].iloc[-4:].values
    
    # 递归预测逻辑
    for i in range(len(future_df)):
        # 处理滞后特征
        if i < 2:  # 前1小时（2个30分钟间隔）
            future_df.loc[future_df.index[i], 'RRP_lag2'] = last_4_steps[-2 + i, 0]
            future_df.loc[future_df.index[i], 'Demand_lag2'] = last_4_steps[-2 + i, 1]
            future_df.loc[future_df.index[i], 'RRP_lag4'] = last_4_steps[-4 + i, 0]
            future_df.loc[future_df.index[i], 'Demand_lag4'] = last_4_steps[-4 + i, 1]
        else:
            # 动态更新滞后特征
            future_df.loc[future_df.index[i], 'RRP_lag2'] = future_df.loc[future_df.index[i-2], 'SpotPrice_RRP']
            future_df.loc[future_df.index[i], 'Demand_lag2'] = future_df.loc[future_df.index[i-2], 'TotalDemand']
            future_df.loc[future_df.index[i], 'RRP_lag4'] = future_df.loc[future_df.index[i-4], 'SpotPrice_RRP']
            future_df.loc[future_df.index[i], 'Demand_lag4'] = future_df.loc[future_df.index[i-4], 'TotalDemand']

        # 执行实时预测
        if i >= 2:  # 当有足够滞后数据时开始预测
            current_features = future_df.loc[future_df.index[i], features].values.reshape(1, -1)
            current_pred = model.predict(scaler_X.transform(current_features))
            unscaled_pred = scaler_y.inverse_transform(current_pred)
            
            # 更新当前时刻的预测值
            future_df.loc[future_df.index[i], 'SpotPrice_RRP'] = unscaled_pred[0, 0]
            future_df.loc[future_df.index[i], 'TotalDemand'] = unscaled_pred[0, 1]
    # ============== 补齐的特征处理代码结束 ============== <!--diff-->
    
    future_X = scaler_X.transform(future_df[features])
    future_pred_scaled = model.predict(future_X)
    future_pred = scaler_y.inverse_transform(future_pred_scaled)
    
    prediction_results = pd.DataFrame({
        'DateTime': future_dates,
        'SpotPrice/RRP': future_pred[:,0],
        'TotalDemand': future_pred[:,1]
    })
    prediction_results.to_excel(writer, sheet_name='预测结果', index=False)
    
    # 保存评估指标
    for feature in metrics_results:
        pd.DataFrame.from_dict(metrics_results[feature], orient='index', columns=['值'])\
            .to_excel(writer, sheet_name=f'{feature}指标')

# 7. 可视化
plt.figure(figsize=(18, 12))

# 电价预测可视化
plt.subplot(2, 1, 1)
plt.plot(prediction_results['DateTime'], prediction_results['SpotPrice/RRP'], label='预测值')
plt.title('电价预测 (2024-04-11)')
plt.ylabel('电价')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.grid(True)

# 负荷预测可视化
plt.subplot(2, 1, 2)
plt.plot(prediction_results['DateTime'], prediction_results['TotalDemand'], label='预测值', color='orange')
plt.title('负荷预测 (2024-04-11)')
plt.ylabel('负荷')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.grid(True)

plt.tight_layout()
plt.savefig('预测结果可视化.png')
plt.show()

print("流程执行完成！预测结果和评估指标已保存到Excel文件。")