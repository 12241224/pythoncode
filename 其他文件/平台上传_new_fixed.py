import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ElectricityPricePredictorSklearn:
    """与原始脚本数据处理等价，但使用 sklearn 随机森林作为回归器。

    保存的 model.pkl 包含 dict: {'model': model, 'price_scaler': price_scaler, 'feature_scaler': feature_scaler,
    'lookback': lookback, 'forecast_horizon': forecast_horizon}
    """

    def __init__(self, lookback=336, forecast_horizon=168, n_estimators=50, random_state=10):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.price_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.model = None
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _get_time_column(self, data):
        """安全地从 DataFrame 中获取时间序列：
        优先查找常见列名（time, timestamp, datetime, date），
        然后尝试使用索引（如果为 datetime 类型），
        再尝试解析任意可解析为 datetime 的列，最后回退为等间隔小时序列。
        返回一个可以传入 pd.to_datetime 的数组/Index。
        """
        candidates = ['time', 'timestamp', 'datetime', 'date', 'Date', 'date_time']
        for c in candidates:
            if c in data.columns:
                return data[c].values

        # 如果索引是时间类型，使用索引
        try:
            index_dtype = getattr(data.index, "dtype", None)
            if pd.api.types.is_datetime64_any_dtype(data.index) or isinstance(index_dtype, pd.DatetimeTZDtype):
                return data.index.to_numpy()
        except Exception:
            pass

        # 尝试解析任意一列为 datetime
        for col in data.columns:
            try:
                parsed = pd.to_datetime(data[col], errors='raise')
                return parsed.values
            except Exception:
                continue

        # 回退：生成等间隔小时序列，长度与行数一致
        return pd.date_range(start=pd.Timestamp('1970-01-01'), periods=len(data), freq='H').to_numpy()

    def _get_column(self, data, target_name, candidates=None, required=True):
        """从 DataFrame 中安全获取列。

        参数:
            data: pd.DataFrame
            target_name: 逻辑名称（用于错误信息），例如 'price'
            candidates: 可选的候选列名列表，按优先级查找
            required: 若为 True 且没找到则抛出 KeyError

        返回 numpy 数组
        """
        if candidates is None:
            candidates = [target_name]

        # 直接按候选名查找
        for c in candidates:
            if c in data.columns:
                return data[c].values

        # 尝试不区分大小写的精确匹配
        lower_map = {col.lower(): col for col in data.columns}
        if target_name.lower() in lower_map:
            return data[lower_map[target_name.lower()]].values

        # 尝试包含关键词的列名
        for col in data.columns:
            if target_name.lower() in col.lower():
                return data[col].values

        if required:
            raise KeyError(f"Required column '{target_name}' not found in CSV. Available columns: {list(data.columns)}")

        # 返回全零作为最后回退（不常用）
        return np.zeros(len(data), dtype=np.float32)

    def create_sequences(self, price_data, feature_data):
        X_price, X_features, y = [], [], []

        if len(price_data) <= self.lookback + self.forecast_horizon:
            raise ValueError("Not enough data to create sequences.")

        for i in range(len(price_data) - self.lookback - self.forecast_horizon + 1):
            price_seq = price_data[i:(i + self.lookback)]
            feature_seq = feature_data[i:(i + self.lookback)]
            target = price_data[(i + self.lookback):(i + self.lookback + self.forecast_horizon)]

            if len(price_seq) == self.lookback and len(target) == self.forecast_horizon:
                X_price.append(price_seq)
                X_features.append(feature_seq)
                y.append(target)

        X_price = np.array(X_price)
        X_features = np.array(X_features)
        y = np.array(y)

        # 将输入两个分支合并为二维特征: flatten time dimension
        # X_price shape -> (n_samples, lookback, 1) -> flatten to (n_samples, lookback)
        X_price_flat = X_price.reshape((X_price.shape[0], X_price.shape[1]))

        # X_features shape -> (n_samples, lookback, n_features) ->  flatten to (n_samples, lookback * n_features)
        X_features_flat = X_features.reshape((X_features.shape[0], X_features.shape[1] * X_features.shape[2]))

        X = np.concatenate([X_price_flat, X_features_flat], axis=1)

        # y currently shape (n_samples, forecast_horizon, 1) or (n_samples, forecast_horizon)
        y_flat = y.reshape((y.shape[0], y.shape[1]))

        return X, y_flat

    def prepare_data(self, time_data, price_data, total_demand, temperature, wind_speed, cloud, humidity, rain):
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

        time_data = time_data[:min_length]
        price_data = price_data[:min_length]
        total_demand = total_demand[:min_length]
        temperature = temperature[:min_length]
        wind_speed = wind_speed[:min_length]
        cloud = cloud[:min_length]
        humidity = humidity[:min_length]
        rain = rain[:min_length]

        hour = np.array(pd.to_datetime(time_data).hour, dtype=np.float32) / 24.0
        day_of_week = np.array(pd.to_datetime(time_data).dayofweek, dtype=np.float32) / 7.0
        month = np.array(pd.to_datetime(time_data).month, dtype=np.float32) / 12.0

        price_scaled = self.price_scaler.fit_transform(price_data.reshape(-1, 1))

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

        # create sequences using scaled arrays
        X_price, X_features, y_seq = self._create_sequences_for_internal(price_scaled, features_scaled)

        return X_price, X_features, y_seq

    def _create_sequences_for_internal(self, price_scaled, features_scaled):
        # price_scaled shape (T,1); features_scaled shape (T, n_features)
        price_seq_list = []
        features_seq_list = []
        y_list = []

        T = price_scaled.shape[0]
        for i in range(T - self.lookback - self.forecast_horizon + 1):
            p_seq = price_scaled[i:i + self.lookback].reshape(self.lookback, 1)
            f_seq = features_scaled[i:i + self.lookback]
            target = price_scaled[i + self.lookback:i + self.lookback + self.forecast_horizon].reshape(self.forecast_horizon)

            price_seq_list.append(p_seq)
            features_seq_list.append(f_seq)
            y_list.append(target)

        X_price = np.array(price_seq_list)
        X_features = np.array(features_seq_list)
        y = np.array(y_list)

        return X_price, X_features, y

    def fit(self, data, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_estimators

        time_data = self._get_time_column(data)
        # 使用更稳健的列获取方式，兼容不同命名
        price_data = self._get_column(
            data,
            'price',
            candidates=['price', 'Price', 'spot_price', 'price_spot', '电价($/MWh)', '电价', '价格', '电价(元/MWh)']
        )
        total_demand = self._get_column(
            data,
            'total_demand',
            candidates=['total_demand', 'demand', 'totalDemand', '总需求负荷（MW）', '总需求负荷', '负荷', '总负荷']
        )
        temperature = self._get_column(
            data,
            'temperature',
            candidates=['temperature', 'temp', 'air_temperature', '温度（摄氏度）', '温度', '气温']
        )
        wind_speed = self._get_column(
            data,
            'wind_speed',
            candidates=['wind_speed', 'windSpeed', 'wind_speed_m/s', '风速（米/秒）', '风速', '风速(m/s)']
        )
        cloud = self._get_column(
            data,
            'cloud',
            candidates=['cloud', 'cloud_cover', 'cloud_cover_pct', '云量（%）', '云量', '云量(%)']
        )
        humidity = self._get_column(
            data,
            'humidity',
            candidates=['humidity', 'relative_humidity', '湿度（%）', '湿度', '相对湿度']
        )
        rain = self._get_column(
            data,
            'rain',
            candidates=['rain', 'precipitation', 'rainfall', '降水量（mm）', '降水量', '降雨量', '降水量(mm)']
        )

        X_price, X_features, y = self.prepare_data(time_data, price_data, total_demand, temperature, wind_speed, cloud, humidity, rain)

        # prepare_data 已经返回按样本分割后的序列：
        # X_price shape -> (n_samples, lookback, 1)
        # X_features shape -> (n_samples, lookback, n_features)
        # y shape -> (n_samples, forecast_horizon)
        # 将它们展平并合并为训练矩阵
        if X_price.ndim != 3 or X_features.ndim != 3:
            raise ValueError("Unexpected sequence shapes from prepare_data: X_price and X_features must be 3D arrays")

        X_price_flat = X_price.reshape((X_price.shape[0], X_price.shape[1]))
        X_features_flat = X_features.reshape((X_features.shape[0], X_features.shape[1] * X_features.shape[2]))
        X = np.concatenate([X_price_flat, X_features_flat], axis=1)
        y_flat = y

        # train/test split
        train_size = int(len(X) * 0.8)
        train_size = max(1, min(train_size, len(X) - 1))

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y_flat[:train_size], y_flat[train_size:]

        # 用随机森林进行多输出回归
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=self.random_state, n_jobs=-1)
        rf.fit(X_train, y_train)

        self.model = rf

        # 返回简单的评估指标
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test) if len(X_test) > 0 else None
        return {'train_score': train_score, 'test_score': test_score}

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        time_data = self._get_time_column(data)
        price_data = self._get_column(
            data,
            'price',
            candidates=['price', 'Price', 'spot_price', 'price_spot', '电价($/MWh)', '电价', '价格', '电价(元/MWh)']
        )
        total_demand = self._get_column(
            data,
            'total_demand',
            candidates=['total_demand', 'demand', 'totalDemand', '总需求负荷（MW）', '总需求负荷', '负荷', '总负荷']
        )
        temperature = self._get_column(
            data,
            'temperature',
            candidates=['temperature', 'temp', 'air_temperature', '温度（摄氏度）', '温度', '气温']
        )
        wind_speed = self._get_column(
            data,
            'wind_speed',
            candidates=['wind_speed', 'windSpeed', 'wind_speed_m/s', '风速（米/秒）', '风速', '风速(m/s)']
        )
        cloud = self._get_column(
            data,
            'cloud',
            candidates=['cloud', 'cloud_cover', 'cloud_cover_pct', '云量（%）', '云量', '云量(%)']
        )
        humidity = self._get_column(
            data,
            'humidity',
            candidates=['humidity', 'relative_humidity', '湿度（%）', '湿度', '相对湿度']
        )
        rain = self._get_column(
            data,
            'rain',
            candidates=['rain', 'precipitation', 'rainfall', '降水量（mm）', '降水量', '降雨量', '降水量(mm)']
        )

        min_length = min(len(time_data), len(price_data), len(total_demand), len(temperature), len(wind_speed), len(cloud), len(humidity), len(rain))
        if min_length < self.lookback:
            raise ValueError("Not enough historical points for prediction.")

        time_data = time_data[:min_length]
        price_data = price_data[:min_length]
        total_demand = total_demand[:min_length]
        temperature = temperature[:min_length]
        wind_speed = wind_speed[:min_length]
        cloud = cloud[:min_length]
        humidity = humidity[:min_length]
        rain = rain[:min_length]

        hour = np.array(pd.to_datetime(time_data).hour, dtype=np.float32) / 24.0
        day_of_week = np.array(pd.to_datetime(time_data).dayofweek, dtype=np.float32) / 7.0
        month = np.array(pd.to_datetime(time_data).month, dtype=np.float32) / 12.0

        price_scaled = self.price_scaler.transform(price_data.reshape(-1, 1))

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

        X_price = price_scaled[-self.lookback:].reshape(1, self.lookback, 1)
        X_features = features_scaled[-self.lookback:].reshape(1, self.lookback, features_scaled.shape[1])

        X_price_flat = X_price.reshape((1, self.lookback))
        X_features_flat = X_features.reshape((1, self.lookback * X_features.shape[2]))
        X = np.concatenate([X_price_flat, X_features_flat], axis=1)

        prediction = self.model.predict(X)  # shape (1, forecast_horizon)

        # inverse scale
        pred_reshaped = prediction.reshape(-1, 1)
        inverse_prediction = self.price_scaler.inverse_transform(pred_reshaped)[:, 0]

        return inverse_prediction

    def save(self, path: str):
        payload = {
            'model': self.model,
            'price_scaler': self.price_scaler,
            'feature_scaler': self.feature_scaler,
            'lookback': self.lookback,
            'forecast_horizon': self.forecast_horizon
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f)




if __name__ == '__main__':
   
    df = pd.read_excel("C:\\Users\\admin\\Desktop\\负荷预测数据 (6).xlsx")
    # 使用周前预测：forecast_horizon=168 表示预测未来 1 周（168 小时）
    predictor = ElectricityPricePredictorSklearn(lookback=336, forecast_horizon=168)  # 使用固定的 lookback 和 weekly forecast_horizon
    predictor.fit(df,n_estimators=50)
    model = predictor
