import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate, Add
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import dill
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate, Add
from tensorflow.keras.optimizers import Adam

class ElectricityPricePredictor:
    def __init__(self, lookback=168, forecast_horizon=168):  # 7天 * 24小时
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.price_scaler = StandardScaler()  # 使用StandardScaler for价格
        self.feature_scaler = MinMaxScaler()  # 使用MinMaxScaler for其他特征
        self.model = None

    def create_sequences(self, price_data, feature_data):
        """创建时序序列数据，分别处理价格和其他特征"""
        X_price, X_features, y = [], [], []

        if len(price_data) <= self.lookback + self.forecast_horizon:
            raise ValueError("Not enough data to create sequences.")

        for i in range(len(price_data) - self.lookback - self.forecast_horizon + 1):
            # 输入序列
            price_seq = price_data[i:(i + self.lookback)]
            feature_seq = feature_data[i:(i + self.lookback)]
            # 目标序列
            target = price_data[(i + self.lookback):(i + self.lookback + self.forecast_horizon)]

            if len(price_seq) == self.lookback and len(target) == self.forecast_horizon:
                X_price.append(price_seq)
                X_features.append(feature_seq)
                y.append(target)

        return np.array(X_price), np.array(X_features), np.array(y)

    def prepare_data(self, time_data, price_data, total_demand, battery, biomass, black_coal, gas, hydro, liquid_fuel, solar, wind):
        """准备训练数据，分别处理价格和其他特征"""
        # 确保输入数据是numpy数组
        price_data = np.array(price_data, dtype=np.float32)
        total_demand = np.array(total_demand, dtype=np.float32)
        battery = np.array(battery, dtype=np.float32)
        biomass = np.array(biomass, dtype=np.float32)
        black_coal = np.array(black_coal, dtype=np.float32)
        gas = np.array(gas, dtype=np.float32)
        hydro = np.array(hydro, dtype=np.float32)
        liquid_fuel = np.array(liquid_fuel, dtype=np.float32)
        solar = np.array(solar, dtype=np.float32)
        wind = np.array(wind, dtype=np.float32)
        min_length = min(len(time_data), len(price_data), len(total_demand), len(battery), len(biomass), len(black_coal), len(gas), len(hydro), len(liquid_fuel), len(solar), len(wind))
        if min_length < (self.lookback + self.forecast_horizon):
            raise ValueError(f"Need at least {self.lookback + self.forecast_horizon} points.")

        # 截取数据
        time_data = time_data[:min_length]
        price_data = price_data[:min_length]
        total_demand = total_demand[:min_length]
        battery = battery[:min_length]
        biomass = biomass[:min_length]
        black_coal = black_coal[:min_length]
        gas = gas[:min_length]
        hydro = hydro[:min_length]
        liquid_fuel = liquid_fuel[:min_length]
        solar = solar[:min_length]
        wind = wind[:min_length]
        # 添加时间特征
        hour = np.array(pd.to_datetime(time_data).hour, dtype=np.float32) / 24.0
        day_of_week = np.array(pd.to_datetime(time_data).dayofweek, dtype=np.float32) / 7.0
        month = np.array(pd.to_datetime(time_data).month, dtype=np.float32) / 12.0

        # 标准化价格数据
        price_scaled = self.price_scaler.fit_transform(price_data.reshape(-1, 1))

        # 标准化其他特征
        features = np.column_stack([
            total_demand,
            battery,
            biomass,
            black_coal,
            gas,
            hydro,
            liquid_fuel,
            solar,
            wind,
            hour,
            day_of_week,
            month
        ])
        features_scaled = self.feature_scaler.fit_transform(features)

        # 创建序列
        X_price, X_features, y = self.create_sequences(price_scaled, features_scaled)

        # 确保数据维度正确
        X_price = X_price.reshape(X_price.shape[0], X_price.shape[1], 1)

        # 划分训练集和测试集
        train_size = int(len(X_price) * 0.8)
        train_size = max(1, min(train_size, len(X_price) - 1))

        return (X_price[:train_size], X_features[:train_size], y[:train_size],
                X_price[train_size:], X_features[train_size:], y[train_size:])

    def build_model(self, price_input_shape, feature_input_shape):
        """构建改进的模型结构"""
        # 价格输入分支
        price_input = Input(shape=price_input_shape)
        price_lstm = Bidirectional(LSTM(128, return_sequences=True))(price_input)
        price_lstm = Dropout(0.2)(price_lstm)
        price_lstm = LSTM(64)(price_lstm)

        # 特征输入分支
        feature_input = Input(shape=feature_input_shape)
        feature_lstm = Bidirectional(LSTM(64, return_sequences=True))(feature_input)
        feature_lstm = Dropout(0.2)(feature_lstm)
        feature_lstm = LSTM(32)(feature_lstm)

        # 合并两个分支
        merged = Concatenate()([price_lstm, feature_lstm])
        dense1 = Dense(256, activation='relu')(merged)
        dense1 = Dropout(0.2)(dense1)
        dense2 = Dense(128, activation='relu')(dense1)

        # 添加残差连接
        output = Dense(self.forecast_horizon)(dense2)

        # 创建模型
        model = Model(inputs=[price_input, feature_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        return model

    def fit(self, data, epochs=100, batch_size=32):
        """训练模型"""
        try:
            # 从DataFrame中提取数据
            time_data = data['time'].values
            price_data = data['price'].values
            total_demand = data['total_demand'].values
            battery = data['battery'].values
            biomass = data['biomass'].values
            black_coal = data['black_coal'].values
            gas = data['gas'].values
            hydro = data['hydro'].values
            liquid_fuel = data['liquid_fuel'].values
            solar = data['solar'].values
            wind = data['wind'].values

            # 准备数据
            train_data = self.prepare_data(time_data, price_data, total_demand, battery, biomass, black_coal, gas, hydro, liquid_fuel, solar, wind)
            X_price_train, X_features_train, y_train, X_price_test, X_features_test, y_test = train_data

            # 构建模型
            self.model = self.build_model(
                price_input_shape=(X_price_train.shape[1], 1),
                feature_input_shape=(X_features_train.shape[1], X_features_train.shape[2])
            )

            # 训练模型
            history = self.model.fit(
                [X_price_train, X_features_train], y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=([X_price_test, X_features_test], y_test),
                verbose=1
            )

            return history

        except Exception as e:
            raise ValueError(f"Error during training: {str(e)}")

    def predict(self, data):
        """进行预测"""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        try:
            # 从DataFrame中提取数据
            time_data = data['time'].values
            price_data = data['price'].values
            total_demand = data['total_demand'].values
            battery = data['battery'].values
            biomass = data['biomass'].values
            black_coal = data['black_coal'].values
            gas = data['gas'].values
            hydro = data['hydro'].values
            liquid_fuel = data['liquid_fuel'].values
            solar = data['solar'].values
            wind = data['wind'].values

            # 准备预测数据
            hour = np.array(pd.to_datetime(time_data).hour, dtype=np.float32) / 24.0
            day_of_week = np.array(pd.to_datetime(time_data).dayofweek, dtype=np.float32) / 7.0
            month = np.array(pd.to_datetime(time_data).month, dtype=np.float32) / 12.0

            # 标准化价格数据
            price_scaled = self.price_scaler.transform(price_data.reshape(-1, 1))

            # 标准化其他特征
            features = np.column_stack([
                total_demand,
                battery,
                biomass,
                black_coal,
                gas,
                hydro,
                liquid_fuel,
                solar,
                wind,
                hour,
                day_of_week,
                month
            ])
            features_scaled = self.feature_scaler.transform(features)

            # 准备预测输入
            X_price = price_scaled[-self.lookback:].reshape(1, self.lookback, 1)
            X_features = features_scaled[-self.lookback:].reshape(1, self.lookback, features_scaled.shape[1])

            # 预测
            prediction = self.model.predict([X_price, X_features])

            # 反向转换预测结果
            prediction_reshaped = prediction.reshape(-1, 1)
            inverse_prediction = self.price_scaler.inverse_transform(prediction_reshaped)[:, 0]

            return inverse_prediction

        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")

    def plot_results(self, history):
        """绘制训练结果"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # 绘制损失
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss During Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # 绘制MAE
        ax2.plot(history.history['mae'], label='Training MAE')
        ax2.plot(history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE During Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_predictions(self, actual_values, predicted_values, time_index):
        """绘制预测结果对比图"""
        plt.figure(figsize=(15, 6))
        plt.plot(time_index, actual_values, label='Actual Price', alpha=0.7)
        plt.plot(time_index, predicted_values, label='Predicted Price', alpha=0.7)
        plt.title('Electricity Price Prediction vs Actual')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

# 使用示例
if __name__ == "__main__":
    try:
        # 生成示例数据 (增加到90天的数据)
        n_samples = 20 * 24  # 90天 * 24小时
        time_data = pd.date_range(start='2023-01-01', periods=n_samples, freq='h')

        # 生成时间模式
        hour_pattern = np.sin(2 * np.pi * (np.array(time_data.hour) / 24))
        week_pattern = np.sin(2 * np.pi * (np.array(time_data.dayofweek) / 7))
        month_pattern = np.sin(2 * np.pi * (np.array(time_data.day) / 30))
        month_trend = 0.1 * np.arange(n_samples)  # 添加长期趋势

        # 生成电价数据
        base_price = 50
        volatility = 0.2
        price_data = (base_price +
                     15 * hour_pattern +  # 日内波动
                     10 * week_pattern +  # 周内波动
                     5 * month_pattern +  # 月内波动
                     month_trend +  # 长期趋势
                     np.random.normal(0, base_price * volatility, n_samples))  # 随机波动

        # 生成各种发电数据
        base_generation = 100
        solar_pattern = np.where(  # 模拟太阳能发电模式
            (time_data.hour >= 6) & (time_data.hour <= 18),  # 白天时间
            np.sin(np.pi * (time_data.hour - 6) / 12),  # 日出到日落的曲线
            0  # 夜间无发电
        )

        # 生成各种发电数据
        total_demand = base_generation + np.random.normal(0, 10, n_samples)
        battery = 20 * np.random.random(n_samples)  # 电池储能
        biomass = 15 + 5 * np.random.random(n_samples)  # 生物质发电
        black_coal = 30 + 10 * np.random.random(n_samples)  # 煤炭发电
        gas = 25 + 8 * np.random.random(n_samples)  # 天然气发电
        hydro = 20 + 5 * np.random.random(n_samples)  # 水力发电
        liquid_fuel = 10 + 3 * np.random.random(n_samples)  # 液体燃料发电
        solar = 50 * solar_pattern + np.random.normal(0, 5, n_samples)  # 太阳能发电
        wind = 30 + 15 * np.random.random(n_samples)  # 风力发电

        # 创建DataFrame并确保数据为正值
        data = pd.DataFrame({
            'time': time_data,
            'price': np.maximum(price_data, 0),
            'total_demand': np.maximum(total_demand, 0),
            'battery': np.maximum(battery, 0),
            'biomass': np.maximum(biomass, 0),
            'black_coal': np.maximum(black_coal, 0),
            'gas': np.maximum(gas, 0),
            'hydro': np.maximum(hydro, 0),
            'liquid_fuel': np.maximum(liquid_fuel, 0),
            'solar': np.maximum(solar, 0),
            'wind': np.maximum(wind, 0)
        })

        print("数据示例：")
        print(data.head())
        print("\n数据统计：")
        print(data.describe())

        # 创建并训练模型
        predictor = ElectricityPricePredictor(lookback=168, forecast_horizon=168)  # 使用固定的lookback和forecast_horizon
        history = predictor.fit(data, epochs=30)

        # 直接使用dill保存预测器实例
        save_path = 'predictor.dill'
        with open(save_path, 'wb') as f:
            dill.dump(predictor, f)
        print(f"预测器已保存到: {save_path}")

        # 直接使用dill加载预测器实例
        with open(save_path, 'rb') as f:
            load_predictor = dill.load(f)
        print("预测器加载成功！")

        # 进行预测
        predictions = load_predictor.predict(data)

        # 创建未来时间索引
        future_dates = pd.date_range(start=time_data[-1] + pd.Timedelta(hours=1),
                                   periods=168, freq='h')

        # 绘制预测结果
        load_predictor.plot_predictions(data['price'].values[-168:], predictions, future_dates)

        # 输出预测结果统计信息
        print("\n预测结果统计：")
        print(f"预测值平均电价: {np.mean(predictions):.2f}")
        print(f"实际平均电价: {np.mean(data['price'].values[-168:]):.2f}")
        print(f"预测值最大电价: {np.max(predictions):.2f}")
        print(f"预测值最小电价: {np.min(predictions):.2f}")
        print(f"实际最大电价: {np.max(data['price'].values[-168:]):.2f}")
        print(f"实际最小电价: {np.min(data['price'].values[-168:]):.2f}")

    except Exception as e:
        print(f"运行示例时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
