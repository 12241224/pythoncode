
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate, Add
from tensorflow.keras.optimizers import Adam


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

    def _map_columns(self, df):
        """尝试把常见列名映射为内部约定的列名（time, price, total_demand, temperature, wind_speed, cloud, humidity, rain）"""
        df = df.copy()
        cols_lower = {c.lower(): c for c in df.columns}

        candidates = {
            'time': ['time', 'datetime', 'date', 'timestamp', 'date_time'],
            'price': ['price', 'spotprice_rrp', 'spot_price', 'spotprice', 'spot_price_rrp', 'spotprice_rrp'],
            'total_demand': ['total_demand', 'totaldemand', 'demand', 'load', 'total_load'],
            'temperature': ['temperature', 'temp'],
            'wind_speed': ['wind_speed', 'windspeed', 'wind'],
            'cloud': ['cloud', 'clouds'],
            'humidity': ['humidity', 'hum'],
            'rain': ['rain', 'precipitation']
        }

        for target, alts in candidates.items():
            # 如果已经存在目标列名，跳过
            if target in df.columns:
                continue
            found = None
            for alt in alts:
                if alt in cols_lower:
                    found = cols_lower[alt]
                    break
            if found is not None:
                df[target] = df[found]

        # 检查是否仍有缺失的必需列
        required = ['time', 'price', 'total_demand', 'temperature', 'wind_speed', 'cloud', 'humidity', 'rain']
        missing = [r for r in required if r not in df.columns]
        if missing:
            raise ValueError(
                f"缺少必要列: {missing}. 数据集包含的列: {list(df.columns)}.\n"
                "请检查列名或在调用前重命名列。"
            )

        return df

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

    def prepare_data(self, time_data, price_data, total_demand, temperature, wind_speed, cloud, humidity, rain):
        """准备训练数据，分别处理价格和其他特征"""
        # 确保输入数据是numpy数组
        price_data = np.array(price_data, dtype=np.float32)
        total_demand = np.array(total_demand, dtype=np.float32)
        temperature = np.array(temperature, dtype=np.float32)
        wind_speed = np.array(wind_speed, dtype=np.float32)
        cloud = np.array(cloud, dtype=np.float32)
        humidity = np.array(humidity, dtype=np.float32)
        rain = np.array(rain, dtype=np.float32)

        min_length = min(len(time_data), len(price_data), len(total_demand), len(temperature), len(wind_speed), len(cloud), len(humidity), len(rain))
        if min_length < (self.lookback + self.forecast_horizon):
            raise ValueError(f"Need at least {self.lookback + self.forecast_horizon} points.")

        # 截取数据
        time_data = time_data[:min_length]
        price_data = price_data[:min_length]
        total_demand = total_demand[:min_length]
        temperature = temperature[:min_length]
        wind_speed = wind_speed[:min_length]
        cloud = cloud[:min_length]
        humidity = humidity[:min_length]
        rain = rain[:min_length]
        # 添加时间特征
        hour = np.array(pd.to_datetime(time_data).hour, dtype=np.float32) / 24.0
        day_of_week = np.array(pd.to_datetime(time_data).dayofweek, dtype=np.float32) / 7.0
        month = np.array(pd.to_datetime(time_data).month, dtype=np.float32) / 12.0

        # 标准化价格数据
        price_scaled = self.price_scaler.fit_transform(price_data.reshape(-1, 1))

        # 标准化其他特征
        features = np.column_stack([
            total_demand,
            temperature,
            wind_speed,
            cloud,
            humidity,
            rain,
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
            # 先做列名映射，兼容不同数据源列名
            data = self._map_columns(data)

            # 从DataFrame中提取数据
            time_data = data['time'].values
            price_data = data['price'].values
            total_demand = data['total_demand'].values
            temperature = data['temperature'].values
            wind_speed = data['wind_speed'].values
            cloud = data['cloud'].values
            humidity = data['humidity'].values
            rain = data['rain'].values

            # 准备数据
            train_data = self.prepare_data(time_data, price_data, total_demand, temperature, wind_speed, cloud, humidity, rain)
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
            # 先做列名映射，兼容不同数据源列名
            data = self._map_columns(data)

            # 从DataFrame中提取数据
            time_data = data['time'].values
            price_data = data['price'].values
            total_demand = data['total_demand'].values
            temperature = data['temperature'].values
            wind_speed = data['wind_speed'].values
            cloud = data['cloud'].values
            humidity = data['humidity'].values
            rain = data['rain'].values

            # 准备预测数据
            hour = np.array(pd.to_datetime(time_data).hour, dtype=np.float32) / 24.0
            day_of_week = np.array(pd.to_datetime(time_data).dayofweek, dtype=np.float32) / 7.0
            month = np.array(pd.to_datetime(time_data).month, dtype=np.float32) / 12.0

            # 标准化价格数据
            price_scaled = self.price_scaler.transform(price_data.reshape(-1, 1))

            # 标准化其他特征
            features = np.column_stack([
                total_demand,
                temperature,
                wind_speed,
                cloud,
                humidity,
                rain,
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

if __name__ == '__main__':
    predictor = ElectricityPricePredictor(lookback=168 * 2, forecast_horizon=168 * 2)  # 使用固定的lookback和forecast_horizon

    # 尝试从几个常见路径加载数据文件（请根据实际情况修改路径或将文件命名为 data.csv）
    import os
    possible_paths = [r"C:\Users\admin\Desktop\实际数据.csv", 'data.csv', 'dataset.csv']
    data = None
    for p in possible_paths:
        if os.path.exists(p):
            data = pd.read_csv(p)
            print(f"Loaded data from {p}")
            break

    if data is None:
        raise FileNotFoundError(
            "未找到数据文件。请在脚本同目录放置数据文件并命名为 data.csv，\n"
            "或编辑脚本并将数据路径赋值给变量 `data`，例如：\n"
            "data = pd.read_csv(r'你的文件路径.csv')"
        )

    predictor.fit(data, epochs=30)
    model = predictor
