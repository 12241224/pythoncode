import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SequenceDataset(Dataset):
    def __init__(self, X_price, X_feat, y):
        self.X_price = X_price.astype(np.float32)
        self.X_feat = X_feat.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_price[idx], self.X_feat[idx], self.y[idx]


class DualLSTM(nn.Module):
    def __init__(self, price_input_size, feature_input_size, hidden_price=128, hidden_feat=64, out_dim=1):
        super().__init__()
        # price branch
        self.price_lstm1 = nn.LSTM(input_size=price_input_size, hidden_size=hidden_price, num_layers=1, batch_first=True, bidirectional=True)
        self.price_dropout = nn.Dropout(0.2)
        self.price_lstm2 = nn.LSTM(input_size=hidden_price*2, hidden_size=64, num_layers=1, batch_first=True)

        # feature branch
        self.feat_lstm1 = nn.LSTM(input_size=feature_input_size, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.feat_dropout = nn.Dropout(0.2)
        self.feat_lstm2 = nn.LSTM(input_size=64*2, hidden_size=32, num_layers=1, batch_first=True)

        # combined
        self.fc1 = nn.Linear(64 + 32, 256)
        self.fc_dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x_price, x_feat):
        # x_price: (B, T, 1)
        b, t, _ = x_price.size()
        p_out, _ = self.price_lstm1(x_price)
        p_out = self.price_dropout(p_out)
        p_out, _ = self.price_lstm2(p_out)
        p_last = p_out[:, -1, :]

        f_out, _ = self.feat_lstm1(x_feat)
        f_out = self.feat_dropout(f_out)
        f_out, _ = self.feat_lstm2(f_out)
        f_last = f_out[:, -1, :]

        merged = torch.cat([p_last, f_last], dim=1)
        x = self.relu(self.fc1(merged))
        x = self.fc_dropout(x)
        x = self.relu(self.fc2(x))
        out = self.out(x)
        return out


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
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2, 'NMAE': nmae, 'NRMSE': nrmse, 'IMAPE': imape}


class ElectricityPricePredictorPyTorch:
    def __init__(self, lookback=24, forecast_horizon=1, device=None):
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.price_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.features = ['TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain',
                         'hour','dayofweek','month','day','SpotPrice_RRP_lag1','TotalDemand_lag1']
        self.target = 'SpotPrice_RRP'

    def prepare_data(self, df):
        df = df.copy()
        for col in ['TotalDemand','Temperature','WindSpeed','Cloud','Humidity','Rain','SpotPrice_RRP']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
        df = df.sort_values('DateTime').reset_index(drop=True)
        df['hour'] = df['DateTime'].dt.hour
        df['dayofweek'] = df['DateTime'].dt.dayofweek
        df['month'] = df['DateTime'].dt.month
        df['day'] = df['DateTime'].dt.day
        if 'WeatherType' in df.columns:
            df['WeatherType'] = pd.factorize(df['WeatherType'].astype(str).fillna('nan'))[0]
        df['SpotPrice_RRP_lag1'] = pd.to_numeric(df['SpotPrice_RRP'].shift(1), errors='coerce')
        df['TotalDemand_lag1'] = pd.to_numeric(df['TotalDemand'].shift(1), errors='coerce')
        df = df.dropna().reset_index(drop=True)

        price_data = df[self.target].values.astype(np.float32)
        feature_data = df[self.features].values.astype(np.float32)

        price_scaled = self.price_scaler.fit_transform(price_data.reshape(-1,1)).flatten()
        features_scaled = self.feature_scaler.fit_transform(feature_data)

        # create sequences
        X_price, X_feat, y, last_idx = [], [], [], []
        n = len(price_scaled)
        for i in range(n - self.lookback - self.forecast_horizon + 1):
            X_price.append(price_scaled[i:i+self.lookback])
            X_feat.append(features_scaled[i:i+self.lookback])
            y.append(price_scaled[i+self.lookback:i+self.lookback+self.forecast_horizon])
            last_idx.append(i + self.lookback - 1)
        X_price = np.array(X_price).reshape(-1, self.lookback, 1)
        X_feat = np.array(X_feat)
        y = np.array(y).reshape(-1, self.forecast_horizon)
        last_idx = np.array(last_idx)

        total_rows = len(df)
        train_row_cut = int(total_rows * 0.95)
        train_mask = last_idx < train_row_cut

        X_price_train = X_price[train_mask]
        X_feat_train = X_feat[train_mask]
        y_train = y[train_mask]

        X_price_test = X_price[~train_mask]
        X_feat_test = X_feat[~train_mask]
        y_test = y[~train_mask]
        test_last_idx = last_idx[~train_mask]

        return (X_price_train, X_feat_train, y_train,
                X_price_test, X_feat_test, y_test,
                df, test_last_idx)

    def build_model(self):
        price_input_size = 1
        feature_input_size = len(self.features)
        model = DualLSTM(price_input_size, feature_input_size, hidden_price=128, hidden_feat=64, out_dim=self.forecast_horizon)
        return model

    def fit(self, df, epochs=50, batch_size=64, lr=1e-3):
        X_price_train, X_feat_train, y_train, X_price_test, X_feat_test, y_test, df_proc, test_last_idx = self.prepare_data(df)
        if X_price_train.shape[0] < 1 or X_price_test.shape[0] < 1:
            raise RuntimeError('训练或测试数据不足，请检查 lookback 和数据长度')

        self.model = self.build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.SmoothL1Loss()

        train_ds = SequenceDataset(X_price_train, X_feat_train, y_train)
        test_ds = SequenceDataset(X_price_test, X_feat_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        best_loss = float('inf')
        best_state = None
        patience = 15
        best_epoch = 0

        for epoch in range(1, epochs+1):
            self.model.train()
            train_losses = []
            for xb_p, xb_f, yb in train_loader:
                xb_p = xb_p.to(self.device)
                xb_f = xb_f.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb_p, xb_f)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # validation
            self.model.eval()
            val_losses = []
            preds = []
            trues = []
            with torch.no_grad():
                for xb_p, xb_f, yb in test_loader:
                    xb_p = xb_p.to(self.device)
                    xb_f = xb_f.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb_p, xb_f)
                    val_losses.append(loss_fn(pred, yb).item())
                    preds.append(pred.cpu().numpy())
                    trues.append(yb.cpu().numpy())

            val_loss = np.mean(val_losses) if len(val_losses)>0 else float('nan')
            preds = np.concatenate(preds) if len(preds)>0 else np.array([])
            trues = np.concatenate(trues) if len(trues)>0 else np.array([])
            print(f"Epoch {epoch}\tTrain Loss: {np.mean(train_losses):.4f}\tVal Loss: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = self.model.state_dict()
                best_epoch = epoch
            if epoch - best_epoch >= patience:
                print('Early stopping')
                break

        # load best state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # save model and components
        torch.save(self.model.state_dict(), 'pytorch_template_model.pth')
        with open('pytorch_template_components.dill', 'wb') as f:
            dill.dump({'price_scaler': self.price_scaler, 'feature_scaler': self.feature_scaler, 'features': self.features}, f)

        # predict on test
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_price_test.astype(np.float32)).to(self.device)
            X_feat_t = torch.from_numpy(X_feat_test.astype(np.float32)).to(self.device)
            preds = self.model(X_test_t, X_feat_t).cpu().numpy()

        preds_flat = preds.reshape(-1, self.forecast_horizon)
        preds_inv = self.price_scaler.inverse_transform(preds_flat)[:,0]

        data_test = df_proc.iloc[test_last_idx].copy().reset_index(drop=True)
        data_test['PredictedPrice'] = preds_inv
        data_test[['DateTime','SpotPrice_RRP','PredictedPrice']].to_csv('template_pytorch_predicted_prices.csv', index=False)

        metrics = calculate_metrics(data_test['SpotPrice_RRP'].values, preds_inv)
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['值'])
        metrics_df.index.name = '指标'
        metrics_df.to_excel('template_pytorch_evaluation_metrics.xlsx')

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12,6))
        plt.plot(data_test['DateTime'], data_test['SpotPrice_RRP'], label='实际价格')
        plt.plot(data_test['DateTime'], data_test['PredictedPrice'], label='预测价格')
        plt.legend()
        plt.tight_layout()
        plt.savefig('template_pytorch_price_comparison.png')

        return metrics

    def predict(self, df):
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

        price_data = df_proc['SpotPrice_RRP'].values.astype(np.float32)
        feature_data = df_proc[self.features].values.astype(np.float32)

        price_scaled = self.price_scaler.transform(price_data.reshape(-1,1)).flatten()
        features_scaled = self.feature_scaler.transform(feature_data)

        if len(price_scaled) < self.lookback:
            raise RuntimeError('数据长度不足以做预测')

        X_price = price_scaled[-self.lookback:].reshape(1, self.lookback, 1)
        X_feat = features_scaled[-self.lookback:].reshape(1, self.lookback, features_scaled.shape[1])

        self.model.load_state_dict(torch.load('pytorch_template_model.pth', map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            xp = torch.from_numpy(X_price.astype(np.float32)).to(self.device)
            xf = torch.from_numpy(X_feat.astype(np.float32)).to(self.device)
            pred = self.model(xp, xf).cpu().numpy()
        pred_inv = self.price_scaler.inverse_transform(pred.reshape(-1,1))[:,0]
        return pred_inv


if __name__ == '__main__':
    data = pd.read_csv(r"C:\Users\admin\Desktop\实际数据.csv", skiprows=0,
                       names=['DateTime','SpotPrice_RRP','TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain'])
    predictor = ElectricityPricePredictorPyTorch(lookback=24, forecast_horizon=1)
    metrics = predictor.fit(data, epochs=30, batch_size=64)
    print('训练完成，评估指标：')
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print('模型与预测已保存（pytorch_template_model.pth, template_pytorch_predicted_prices.csv 等）')
