import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor


class ElectricityPricePredictorLGBM:
    """与原始脚本数据处理等价，但使用 LightGBM 多输出回归。

    保存的 model.pkl 包含 dict: {
        'model': model, 'price_scaler': price_scaler, 'feature_scaler': feature_scaler,
        'lookback': lookback, 'forecast_horizon': forecast_horizon
    }
    """

    def __init__(self, lookback=168, forecast_horizon=168, n_estimators=200, random_state=42):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.price_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.model = None
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_time_column(self, data):
        candidates = ['time', 'timestamp', 'datetime', 'date', 'Date', 'date_time']
        for c in candidates:
            if c in data.columns:
                return data[c].values

        try:
            if pd.api.types.is_datetime64_any_dtype(data.index) or pd.api.types.is_datetime64tz_dtype(data.index):
                return data.index.to_numpy()
        except Exception:
            pass

        for col in data.columns:
            try:
                parsed = pd.to_datetime(data[col], errors='raise')
                return parsed.values
            except Exception:
                continue

        return pd.date_range(start=pd.Timestamp('1970-01-01'), periods=len(data), freq='H').to_numpy()

    def _get_column(self, data, target_name, candidates=None, required=True):
        if candidates is None:
            candidates = [target_name]

        for c in candidates:
            if c in data.columns:
                return data[c].values

        lower_map = {col.lower(): col for col in data.columns}
        if target_name.lower() in lower_map:
            return data[lower_map[target_name.lower()]].values

        for col in data.columns:
            if target_name.lower() in col.lower():
                return data[col].values

        if required:
            raise KeyError(f"Required column '{target_name}' not found. Available columns: {list(data.columns)}")

        return np.zeros(len(data), dtype=np.float32)

    def _create_sequences_for_internal(self, price_scaled, features_scaled):
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

        X_price, X_features, y_seq = self._create_sequences_for_internal(price_scaled, features_scaled)
        return X_price, X_features, y_seq

    def fit(self, data, n_estimators=None):
        if n_estimators is None:
            n_estimators = self.n_estimators

        self.logger.info("Starting fit: n_estimators=%s, lookback=%s, forecast_horizon=%s", n_estimators, self.lookback, self.forecast_horizon)

        time_data = self._get_time_column(data)
        price_data = self._get_column(data, 'price', candidates=['price', 'Price', 'spot_price', 'price_spot'])
        total_demand = self._get_column(data, 'total_demand', candidates=['total_demand', 'demand', 'totalDemand'])
        temperature = self._get_column(data, 'temperature', candidates=['temperature', 'temp', 'air_temperature'])
        wind_speed = self._get_column(data, 'wind_speed', candidates=['wind_speed', 'windSpeed', 'wind_speed_m/s'])
        cloud = self._get_column(data, 'cloud', candidates=['cloud', 'cloud_cover', 'cloud_cover_pct'])
        humidity = self._get_column(data, 'humidity', candidates=['humidity', 'relative_humidity'])
        rain = self._get_column(data, 'rain', candidates=['rain', 'precipitation', 'rainfall'])

        X_price, X_features, y = self.prepare_data(time_data, price_data, total_demand, temperature, wind_speed, cloud, humidity, rain)
        self.logger.debug("prepare_data returned shapes: X_price=%s, X_features=%s, y=%s", getattr(X_price, 'shape', None), getattr(X_features, 'shape', None), getattr(y, 'shape', None))

        # flatten and build training matrix
        X_price_flat = X_price.reshape((X_price.shape[0], X_price.shape[1]))
        X_features_flat = X_features.reshape((X_features.shape[0], X_features.shape[1] * X_features.shape[2]))
        X = np.concatenate([X_price_flat, X_features_flat], axis=1)
        y_flat = y

        self.logger.info("Constructed training matrix X shape=%s, y shape=%s", X.shape, y_flat.shape)

        # train/test split
        train_size = int(len(X) * 0.8)
        train_size = max(1, min(train_size, len(X) - 1))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y_flat[:train_size], y_flat[train_size:]
        self.logger.info("Train/test sizes: X_train=%s, X_test=%s", X_train.shape, X_test.shape)

        # LightGBM multi-output via MultiOutputRegressor
        base = LGBMRegressor(n_estimators=n_estimators, random_state=self.random_state)
        mor = MultiOutputRegressor(base, n_jobs=-1)
        self.logger.info("Beginning model.fit (LightGBM) ...")
        mor.fit(X_train, y_train)
        self.logger.info("Model.fit completed")

        self.model = mor

        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test) if len(X_test) > 0 else None
        self.logger.info("Training finished. train_score=%s, test_score=%s", train_score, test_score)
        return {'train_score': train_score, 'test_score': test_score}

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        time_data = self._get_time_column(data)
        price_data = self._get_column(data, 'price', candidates=['price', 'Price', 'spot_price', 'price_spot'])
        total_demand = self._get_column(data, 'total_demand', candidates=['total_demand', 'demand', 'totalDemand'])
        temperature = self._get_column(data, 'temperature', candidates=['temperature', 'temp', 'air_temperature'])
        wind_speed = self._get_column(data, 'wind_speed', candidates=['wind_speed', 'windSpeed', 'wind_speed_m/s'])
        cloud = self._get_column(data, 'cloud', candidates=['cloud', 'cloud_cover', 'cloud_cover_pct'])
        humidity = self._get_column(data, 'humidity', candidates=['humidity', 'relative_humidity'])
        rain = self._get_column(data, 'rain', candidates=['rain', 'precipitation', 'rainfall'])

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

        prediction = self.model.predict(X)

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


def main(csv_path: str, model_out: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logger = logging.getLogger('platform_upload_lightgbm')

    df = pd.read_csv(csv_path)

    predictor = ElectricityPricePredictorLGBM(lookback=168 * 2, forecast_horizon=168 * 2)
    res = predictor.fit(df)
    logger.info('Training result: %s', res)
    predictor.save(model_out)
    logger.info('Model saved to %s', model_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='cleaned_electromagnetic_data.csv', help='训练数据 CSV 路径，需包含 time, price, total_demand, temperature, wind_speed, cloud, humidity, rain 列')
    parser.add_argument('--out', type=str, default='model_lightgbm.pkl', help='输出模型路径 (.pkl)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    main(str(csv_path), args.out)
