import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
class ElectricityPricePredictor:
    def __init__(self):
        """
        初始化电价预测模型
        """
        self.models = {
            'load': RandomForestRegressor(n_estimators=100, random_state=42),
            'wind': RandomForestRegressor(n_estimators=100, random_state=42),
            'hydro': RandomForestRegressor(n_estimators=100, random_state=42),
            'solar': RandomForestRegressor(n_estimators=100, random_state=42),
            'price': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scalers = {
            'input': StandardScaler(),
            'output': StandardScaler()
        }
        self.feature_importances = {}
        
    def preprocess_data(self, data):
        """
        数据预处理
        :param data: 包含历史数据的DataFrame
        :return: 处理后的数据
        """
        # 转换日期时间特征
        data['datetime'] = pd.to_datetime(data['datetime'])
        data['hour'] = data['datetime'].dt.hour
        data['day_of_week'] = data['datetime'].dt.dayofweek
        data['month'] = data['datetime'].dt.month
        data['day_of_year'] = data['datetime'].dt.dayofyear
        
        # 添加季节特征
        data['season'] = data['month'] % 12 // 3 + 1
        
        # 添加是否工作日特征
        data['is_weekday'] = data['day_of_week'].apply(lambda x: 1 if x < 5 else 0)
        
        return data
    
    def train(self, historical_data):
        """
        训练模型
        :param historical_data: 包含历史数据的DataFrame
        """
        # 数据预处理
        data = self.preprocess_data(historical_data)
        
        # 定义特征和目标变量
        common_features = ['hour', 'day_of_week', 'month', 'day_of_year', 'season', 'is_weekday',
                          'temperature', 'humidity', 'wind_speed']
        
        # 训练负荷预测模型
        X_load = data[common_features]
        y_load = data['load']
        self.models['load'].fit(X_load, y_load)
        self.feature_importances['load'] = self.models['load'].feature_importances_
        
        # 训练风力发电预测模型
        X_wind = data[common_features + ['wind_speed', 'wind_direction']]
        y_wind = data['wind_generation']
        self.models['wind'].fit(X_wind, y_wind)
        self.feature_importances['wind'] = self.models['wind'].feature_importances_
        
        # 训练水力发电预测模型
        X_hydro = data[common_features + ['precipitation', 'reservoir_level']]
        y_hydro = data['hydro_generation']
        self.models['hydro'].fit(X_hydro, y_hydro)
        self.feature_importances['hydro'] = self.models['hydro'].feature_importances_
        
        # 训练光伏发电预测模型
        X_solar = data[common_features + ['solar_radiation', 'cloud_cover']]
        y_solar = data['solar_generation']
        self.models['solar'].fit(X_solar, y_solar)
        self.feature_importances['solar'] = self.models['solar'].feature_importances_
        
        # 训练电价预测模型
        X_price = data[common_features + ['load', 'wind_generation', 'hydro_generation', 'solar_generation']]
        y_price = data['price']
        self.models['price'].fit(X_price, y_price)
        self.feature_importances['price'] = self.models['price'].feature_importances_
    
    def predict(self, forecast_data):
        """
        进行预测
        :param forecast_data: 包含预测输入特征的DataFrame
        :return: 包含所有预测结果的字典
        """
        # 数据预处理
        data = self.preprocess_data(forecast_data)
        
        # 定义特征
        common_features = ['hour', 'day_of_week', 'month', 'day_of_year', 'season', 'is_weekday',
                          'temperature', 'humidity', 'wind_speed']
        
        # 预测负荷
        X_load = data[common_features]
        load_pred = self.models['load'].predict(X_load)
        
        # 预测风力发电
        X_wind = data[common_features + ['wind_speed', 'wind_direction']]
        wind_pred = self.models['wind'].predict(X_wind)
        
        # 预测水力发电
        X_hydro = data[common_features + ['precipitation', 'reservoir_level']]
        hydro_pred = self.models['hydro'].predict(X_hydro)
        
        # 预测光伏发电
        X_solar = data[common_features + ['solar_radiation', 'cloud_cover']]
        solar_pred = self.models['solar'].predict(X_solar)
        
        # 预测电价
        X_price = data[common_features]
        X_price['load'] = load_pred
        X_price['wind_generation'] = wind_pred
        X_price['hydro_generation'] = hydro_pred
        X_price['solar_generation'] = solar_pred
        price_pred = self.models['price'].predict(X_price)
        
        # 返回所有预测结果
        predictions = {
            'datetime': data['datetime'],
            'load_pred': load_pred,
            'wind_pred': wind_pred,
            'hydro_pred': hydro_pred,
            'solar_pred': solar_pred,
            'price_pred': price_pred
        }
        
        return predictions
    
    def evaluate(self, test_data):
        """
        评估模型性能
        :param test_data: 包含测试数据的DataFrame
        :return: 评估结果字典
        """
        # 数据预处理
        data = self.preprocess_data(test_data)
        
        # 进行预测
        predictions = self.predict(test_data)
        
        # 计算评估指标
        metrics = {}
        
        # 负荷预测评估
        metrics['load'] = {
            'MAE': mean_absolute_error(data['load'], predictions['load_pred']),
            'RMSE': np.sqrt(mean_squared_error(data['load'], predictions['load_pred']))
        }
        
        # 风力发电预测评估
        metrics['wind'] = {
            'MAE': mean_absolute_error(data['wind_generation'], predictions['wind_pred']),
            'RMSE': np.sqrt(mean_squared_error(data['wind_generation'], predictions['wind_pred']))
        }
        
        # 水力发电预测评估
        metrics['hydro'] = {
            'MAE': mean_absolute_error(data['hydro_generation'], predictions['hydro_pred']),
            'RMSE': np.sqrt(mean_squared_error(data['hydro_generation'], predictions['hydro_pred']))
        }
        
        # 光伏发电预测评估
        metrics['solar'] = {
            'MAE': mean_absolute_error(data['solar_generation'], predictions['solar_pred']),
            'RMSE': np.sqrt(mean_squared_error(data['solar_generation'], predictions['solar_pred']))
        }
        
        # 电价预测评估
        metrics['price'] = {
            'MAE': mean_absolute_error(data['price'], predictions['price_pred']),
            'RMSE': np.sqrt(mean_squared_error(data['price'], predictions['price_pred']))
        }
        
        return metrics
    
    def plot_feature_importance(self, model_type='price'):
        """
        绘制特征重要性图
        :param model_type: 模型类型 ('load', 'wind', 'hydro', 'solar', 'price')
        """
        if model_type not in self.feature_importances:
            print(f"No feature importance data for {model_type}")
            return
            
        importances = self.feature_importances[model_type]
        
        if model_type == 'price':
            features = ['hour', 'day_of_week', 'month', 'day_of_year', 'season', 'is_weekday',
                       'temperature', 'humidity', 'wind_speed', 'load', 'wind_generation', 
                       'hydro_generation', 'solar_generation']
        else:
            features = ['hour', 'day_of_week', 'month', 'day_of_year', 'season', 'is_weekday',
                       'temperature', 'humidity', 'wind_speed']
            if model_type == 'wind':
                features.extend(['wind_speed', 'wind_direction'])
            elif model_type == 'hydro':
                features.extend(['precipitation', 'reservoir_level'])
            elif model_type == 'solar':
                features.extend(['solar_radiation', 'cloud_cover'])
        
        # 创建DataFrame便于排序
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # 绘制条形图
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Feature Importance')
        plt.title(f'Feature Importance for {model_type} Prediction')
        plt.gca().invert_yaxis()
        plt.show()
# 示例数据生成 
def generate_sample_data(days=365):
    dates = pd.date_range(start='2025-01-01', periods=days*24, freq='H')
    data = pd.DataFrame({'datetime': dates})
    
    # 添加基本特征
    data['temperature'] = np.sin(2 * np.pi * data.index / (24*365)) * 15 + 20  # 季节性变化
    data['humidity'] = np.random.normal(50, 10, len(data))
    data['wind_speed'] = np.random.gamma(2, 2, len(data))
    data['wind_direction'] = np.random.randint(0, 360, len(data))
    data['precipitation'] = np.random.poisson(0.1, len(data))
    data['reservoir_level'] = np.linspace(80, 120, len(data)) + np.random.normal(0, 5, len(data))
    data['solar_radiation'] = (np.sin(2 * np.pi * (data['datetime'].dt.hour - 6) / 12) * 800).clip(0)
    data['cloud_cover'] = np.random.randint(0, 100, len(data))
    
    # 生成模拟负荷数据 (有日和周模式)
    daily_pattern = np.sin(2 * np.pi * (data['datetime'].dt.hour - 6) / 24) * 500 + 1000
    weekly_pattern = np.where(data['datetime'].dt.dayofweek < 5, 1.2, 0.8)  # 工作日更高
    seasonal_pattern = np.sin(2 * np.pi * data['datetime'].dt.dayofyear / 365) * 200 + 800
    data['load'] = daily_pattern * weekly_pattern + seasonal_pattern + np.random.normal(0, 50, len(data))
    
    # 生成模拟发电数据
    data['wind_generation'] = data['wind_speed']**2 * 0.5 + np.random.normal(0, 20, len(data))
    data['hydro_generation'] = data['reservoir_level'] * 2 + data['precipitation'] * 10 + np.random.normal(0, 30, len(data))
    data['solar_generation'] = data['solar_radiation'] * 0.4 * (1 - data['cloud_cover'] / 200) + np.random.normal(0, 10, len(data))
    
    # 生成模拟电价 (基于供需关系)
    total_supply = data['wind_generation'] + data['hydro_generation'] + data['solar_generation']
    imbalance = data['load'] - total_supply
    base_price = 50
    data['price'] = base_price + imbalance * 0.01 + np.random.normal(0, 2, len(data))
    
    return data

# 生成并分割数据
full_data = generate_sample_data(365)
train_data, test_data = train_test_split(full_data, test_size=0.2, shuffle=False)

# 初始化并训练模型
predictor = ElectricityPricePredictor()
predictor.train(train_data)

# 在测试集上评估
metrics = predictor.evaluate(test_data)
print("Evaluation Metrics:")
for model_type, metric in metrics.items():
    print(f"\n{model_type.upper()} Prediction:")
    print(f"MAE: {metric['MAE']:.2f}")
    print(f"RMSE: {metric['RMSE']:.2f}")

# 进行预测
future_data = generate_sample_data(7)  # 生成未来一周的数据
predictions = predictor.predict(future_data)

# 可视化电价预测结果
plt.figure(figsize=(12, 6))
plt.plot(predictions['datetime'], predictions['price_pred'], label='Predicted Price')
plt.xlabel('Datetime')
plt.ylabel('Price ($/MWh)')
plt.title('Electricity Price Prediction')
plt.legend()
plt.grid()
plt.show()

# 查看特征重要性
predictor.plot_feature_importance('price')
predictor.plot_feature_importance('wind')
predictor.plot_feature_importance('solar')
predictor.plot_feature_importance('hydro')
predictor.plot_feature_importance('load')

# 进行预测
future_data = generate_sample_data(7)  # 生成未来一周的数据
predictions = predictor.predict(future_data)

# 将预测结果保存到CSV文件
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('predicted_prices.csv', index=False)

# 打印保存路径
print("预测结果已保存到 predicted_prices.csv 文件中")

# 读取CSV文件
predictions_df = pd.read_csv('predicted_prices.csv')

# 显示数据
print(predictions_df)
