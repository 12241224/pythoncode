import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import dill

# 评估函数（与原脚本一致）
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
    r2 = r2_score(y_true, y_pred)
    denom = (np.max(y_true) - np.min(y_true))
    if denom == 0:
        nmae = np.nan
        nrmse = np.nan
    else:
        nmae = mae / denom
        nrmse = rmse / denom
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
                imape_values.append(min(abs((30 - pred_val) / 30), 1))
        else:
            imape_values.append(min(abs((true_val - pred_val) / true_val), 1))
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


class ElectricityPricePredictor:
    def __init__(self, lookback=24, forecast_horizon=1):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.price_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.model = None

    def create_sequences(self, price_data, feature_data):
        X_price, X_features, y, last_idx = [], [], [], []
        n = len(price_data)
        for i in range(n - self.lookback - self.forecast_horizon + 1):
            price_seq = price_data[i:(i + self.lookback)]
            feature_seq = feature_data[i:(i + self.lookback)]
            target = price_data[(i + self.lookback):(i + self.lookback + self.forecast_horizon)]
            if len(price_seq) == self.lookback and len(target) == self.forecast_horizon:
                X_price.append(price_seq)
                X_features.append(feature_seq)
                y.append(target)
                last_idx.append(i + self.lookback - 1)
        return np.array(X_price), np.array(X_features), np.array(y), np.array(last_idx)

    def prepare_data(self, df):
        # df: dataframe containing DateTime, SpotPrice_RRP, TotalDemand, WeatherType, Temperature, WindSpeed, Cloud, Humidity, Rain
        df = df.copy()
        # ensure numeric
        for col in ['TotalDemand','Temperature','WindSpeed','Cloud','Humidity','Rain','SpotPrice_RRP']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # datetime
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df = df.sort_values('DateTime').reset_index(drop=True)
        # time features
        df['hour'] = df['DateTime'].dt.hour
        df['dayofweek'] = df['DateTime'].dt.dayofweek
        df['month'] = df['DateTime'].dt.month
        df['day'] = df['DateTime'].dt.day
        # encode WeatherType
        if 'WeatherType' in df.columns:
            df['WeatherType'] = df['WeatherType'].astype(str).fillna('nan')
            # simple label encoding via pandas factorize to avoid sklearn dependency here
            df['WeatherType'] = pd.factorize(df['WeatherType'])[0]
        # lag features
        df['SpotPrice_RRP_lag1'] = pd.to_numeric(df['SpotPrice_RRP'].shift(1), errors='coerce')
        df['TotalDemand_lag1'] = pd.to_numeric(df['TotalDemand'].shift(1), errors='coerce')
        df = df.dropna().reset_index(drop=True)

        target = 'SpotPrice_RRP'
        features = ['TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain',
                    'hour','dayofweek','month','day','SpotPrice_RRP_lag1','TotalDemand_lag1']

        price_data = df[target].values.astype(np.float32)
        feature_data = df[features].values.astype(np.float32)

        # scale
        price_scaled = self.price_scaler.fit_transform(price_data.reshape(-1,1)).flatten()
        features_scaled = self.feature_scaler.fit_transform(feature_data)

        # sequences and last index
        X_price, X_features, y, last_idx = self.create_sequences(price_scaled, features_scaled)
        # reshape price to (N, lookback, 1)
        X_price = X_price.reshape((X_price.shape[0], X_price.shape[1], 1))
        # y shape -> (N, forecast_horizon)

        # determine train/test split by original row index: use 95% train like original
        total_rows = len(df)
        train_row_cut = int(total_rows * 0.95)
        train_mask = last_idx < train_row_cut

        X_price_train = X_price[train_mask]
        X_features_train = X_features[train_mask]
        y_train = y[train_mask]

        X_price_test = X_price[~train_mask]
        X_features_test = X_features[~train_mask]
        y_test = y[~train_mask]
        test_last_idx = last_idx[~train_mask]

        return (X_price_train, X_features_train, y_train,
                X_price_test, X_features_test, y_test,
                df, test_last_idx)

    def build_model(self, price_input_shape, feature_input_shape):
        price_input = Input(shape=price_input_shape)
        price_lstm = Bidirectional(LSTM(128, return_sequences=True))(price_input)
        price_lstm = Dropout(0.2)(price_lstm)
        price_lstm = LSTM(64)(price_lstm)

        feature_input = Input(shape=feature_input_shape)
        feature_lstm = Bidirectional(LSTM(64, return_sequences=True))(feature_input)
        feature_lstm = Dropout(0.2)(feature_lstm)
        feature_lstm = LSTM(32)(feature_lstm)

        merged = Concatenate()([price_lstm, feature_lstm])
        dense1 = Dense(256, activation='relu')(merged)
        dense1 = Dropout(0.2)(dense1)
        dense2 = Dense(128, activation='relu')(dense1)

        output = Dense(self.forecast_horizon)(dense2)
        model = Model(inputs=[price_input, feature_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        return model

    def fit(self, df, epochs=50, batch_size=64):
        try:
            X_price_train, X_features_train, y_train, X_price_test, X_features_test, y_test, df_processed, test_last_idx = self.prepare_data(df)
            if X_price_train.shape[0] < 1 or X_price_test.shape[0] < 1:
                raise ValueError('训练或测试数据不足，请检查 lookback 和数据长度')

            self.model = self.build_model(
                price_input_shape=(X_price_train.shape[1], 1),
                feature_input_shape=(X_features_train.shape[1], X_features_train.shape[2])
            )

            history = self.model.fit(
                [X_price_train, X_features_train], y_train,
                validation_data=([X_price_test, X_features_test], y_test),
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )

            # save model and scalers
            self.model.save('keras_lstm_price_model.h5')
            with open('keras_lstm_components.dill', 'wb') as f:
                dill.dump({'price_scaler': self.price_scaler, 'feature_scaler': self.feature_scaler, 'features': df_processed.columns.tolist()}, f)

            # predict on test set and save aligned results
            preds = self.model.predict([X_price_test, X_features_test])
            # preds shape (N, forecast_horizon)
            preds_flat = preds.reshape(-1, self.forecast_horizon)
            # inverse transform
            preds_inv = self.price_scaler.inverse_transform(preds_flat)[:,0]

            # build data_test aligned with last_idx positions
            data_test = df_processed.iloc[test_last_idx].copy().reset_index(drop=True)
            data_test['PredictedPrice'] = preds_inv
            data_test[['DateTime','SpotPrice_RRP','PredictedPrice']].to_csv('template_predicted_prices.csv', index=False)

            # metrics
            y_true = data_test['SpotPrice_RRP'].values
            metrics = calculate_metrics(y_true, preds_inv)
            metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['值'])
            metrics_df.index.name = '指标'
            metrics_df.to_excel('template_evaluation_metrics.xlsx')

            # plot
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.figure(figsize=(12,6))
            plt.plot(data_test['DateTime'], data_test['SpotPrice_RRP'], label='实际价格')
            plt.plot(data_test['DateTime'], data_test['PredictedPrice'], label='预测价格')
            plt.legend()
            plt.tight_layout()
            plt.savefig('template_price_comparison.png')

            return history
        except Exception as e:
            raise RuntimeError(f'训练失败: {e}')

    def predict(self, df):
        if self.model is None:
            raise RuntimeError('模型尚未训练')
        # prepare sliding window last lookback
        df_proc = df.copy()
        for col in ['TotalDemand','Temperature','WindSpeed','Cloud','Humidity','Rain','SpotPrice_RRP']:
            if col in df_proc.columns:
                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
        df_proc['DateTime'] = pd.to_datetime(df_proc['DateTime'], errors='coerce')
        df_proc = df_proc.sort_values('DateTime').reset_index(drop=True)
        df_proc['hour'] = df_proc['DateTime'].dt.hour
        df_proc['dayofweek'] = df_proc['DateTime'].dt.dayofweek
        df_proc['month'] = df_proc['DateTime'].dt.month
        df_proc['day'] = df_proc['DateTime'].dt.day
        if 'WeatherType' in df_proc.columns:
            df_proc['WeatherType'] = pd.factorize(df_proc['WeatherType'].astype(str).fillna('nan'))[0]
        df_proc['SpotPrice_RRP_lag1'] = pd.to_numeric(df_proc['SpotPrice_RRP'].shift(1), errors='coerce')
        df_proc['TotalDemand_lag1'] = pd.to_numeric(df_proc['TotalDemand'].shift(1), errors='coerce')
        df_proc = df_proc.dropna().reset_index(drop=True)

        features = ['TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain',
                    'hour','dayofweek','month','day','SpotPrice_RRP_lag1','TotalDemand_lag1']
        price_data = df_proc['SpotPrice_RRP'].values.astype(np.float32)
        feature_data = df_proc[features].values.astype(np.float32)

        price_scaled = self.price_scaler.transform(price_data.reshape(-1,1)).flatten()
        features_scaled = self.feature_scaler.transform(feature_data)

        if len(price_scaled) < self.lookback:
            raise RuntimeError('数据长度不足以做预测')

        X_price = price_scaled[-self.lookback:].reshape(1, self.lookback, 1)
        X_features = features_scaled[-self.lookback:].reshape(1, self.lookback, features_scaled.shape[1])

        pred = self.model.predict([X_price, X_features])
        pred_inv = self.price_scaler.inverse_transform(pred.reshape(-1,1))[:,0]
        return pred_inv


if __name__ == '__main__':
    # 读取数据并训练
    data = pd.read_csv(r"C:\Users\admin\AppData\Roaming\Microsoft\Windows\Recent", skiprows=0,
                       names=['DateTime','SpotPrice_RRP','TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain'])
    predictor = ElectricityPricePredictor(lookback=24, forecast_horizon=1)
    predictor.fit(data, epochs=30, batch_size=64)
    print('训练完成，模型与预测已保存。')


predictor = ElectricityPricePredictor(lookback=168 * 2, forecast_horizon=168 * 2)  # 使用固定的lookback和forecast_horizon
predictor.fit(data, epochs=30)
model = predictor
