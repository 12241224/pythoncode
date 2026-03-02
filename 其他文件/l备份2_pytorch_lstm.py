import sys

try:
    import pandas as pd
except Exception:
    print('缺少 pandas 库，请使用: pip install pandas')
    raise SystemExit(1)

try:
    import numpy as np
except Exception:
    print('缺少 numpy 库，请使用: pip install numpy')
    raise SystemExit(1)

try:
    import torch
    import torch.nn as nn
    import random
    from torch.utils.data import Dataset, DataLoader
except Exception:
    print('缺少 torch 库，请使用: pip install torch （或参照 https://pytorch.org/ 安装）')
    raise SystemExit(1)

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except Exception:
    print('缺少 scikit-learn 库，请使用: pip install scikit-learn')
    raise SystemExit(1)

try:
    import dill
except Exception:
    print('缺少 dill 库，请使用: pip install dill')
    raise SystemExit(1)

# ------------- 公共函数 -------------
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
        if np.isnan(mape):
            mape = np.nan
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


# ------------- Dataset for LSTM -------------
class LSTSDataset(Dataset):
    def __init__(self, X, y=None, seq_len=24):
        # X: 2D array (samples, features)
        self.seq_len = seq_len
        self.X_full = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32)
        self.samples = []
        self.targets = []
        # build sequences sliding window along time (assumes consecutive ordering)
        for i in range(len(self.X_full) - seq_len + 1):
            self.samples.append(self.X_full[i:i+seq_len])
            if self.y is not None:
                # target is the value at the last time step of the window
                self.targets.append(self.y[i+seq_len-1])
        self.samples = np.stack(self.samples)  # (N, seq_len, feat)
        if self.y is not None:
            self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.y is None:
            return self.samples[idx]
        return self.samples[idx], self.targets[idx]


# ------------- LSTM Model -------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, feat)
        out, (hn, cn) = self.lstm(x)
        # take the last output
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


# ------------- main flow -------------
def main(seq_len=24):
    # 数据加载（保持与原脚本一致）
    data = pd.read_csv(r"C:\Users\admin\Desktop\实际数据.csv",
                       skiprows=0,
                       names=['DateTime','SpotPrice_RRP','TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain'])

    for col in ['TotalDemand', 'Temperature', 'WindSpeed', 'Cloud', 'Humidity', 'Rain', 'SpotPrice_RRP']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
    data = data.sort_values('DateTime')

    # 时间特征
    data['hour'] = data['DateTime'].dt.hour
    data['dayofweek'] = data['DateTime'].dt.dayofweek
    data['month'] = data['DateTime'].dt.month
    data['day'] = data['DateTime'].dt.day

    # 编码类别特征
    weather_le = LabelEncoder()
    data['WeatherType'] = weather_le.fit_transform(data['WeatherType'].astype(str))

    # 滞后特征（这里仍计算1阶滞后）
    data['SpotPrice_RRP_lag1'] = pd.to_numeric(data['SpotPrice_RRP'].shift(1), errors='coerce')
    data['TotalDemand_lag1'] = pd.to_numeric(data['TotalDemand'].shift(1), errors='coerce')

    data = data.dropna().reset_index(drop=True)

    target = 'SpotPrice_RRP'
    features = ['TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain',
                'hour','dayofweek','month','day','SpotPrice_RRP_lag1','TotalDemand_lag1']

    X_all = data[features].values
    y_all = data[target].values

    # 划分训练/测试（注意：因为 LSTM 使用序列窗口，边界处理略有区别）
    train_size = int(len(data) * 0.95)
    X_train_full, X_test_full = X_all[:train_size], X_all[train_size:]
    y_train_full, y_test_full = y_all[:train_size], y_all[train_size:]

    # 标准化（fit 用训练集全部样本表征）
    scaler = StandardScaler()
    scaler.fit(X_train_full)
    X_train_scaled = scaler.transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)

    # 构建序列数据集（滑动窗口）
    train_dataset = LSTSDataset(X_train_scaled, y_train_full, seq_len=seq_len)
    # 对测试集也构建滑动窗口，但需要注意最后的对齐：为了比较与原脚本（使用 iloc 切片测试集），
    # 这里我们会把测试段也视为独立连续序列并生成窗口。预测对应窗口的最后一步作为该窗口的预测。
    test_dataset = LSTSDataset(X_test_scaled, y_test_full, seq_len=seq_len)

    # 如果测试样本过少（小于1个窗口），直接退出提示
    if len(test_dataset) == 0:
        print('测试集太小，seq_len={seq_len} 会导致没有可用测试窗口，请减小 seq_len')
        return

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = LSTMRegressor(input_size=X_train_scaled.shape[1], hidden_size=64, num_layers=2, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    best_val_loss = float('inf')
    best_epoch = 0
    epochs = 200
    patience = 15

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # 验证
        model.eval()
        val_losses = []
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())

        val_loss = np.mean(val_losses) if len(val_losses)>0 else float('nan')
        preds = np.concatenate(preds) if len(preds)>0 else np.array([])
        trues = np.concatenate(trues) if len(trues)>0 else np.array([])

        print(f"Epoch {epoch}\tTrain MAE: {np.mean(train_losses):.4f}\tVal MAE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'pytorch_lstm_regressor.pth')
        if epoch - best_epoch >= patience:
            print('Early stopping triggered')
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('pytorch_lstm_regressor.pth', map_location=device))
    model.eval()

    # 在测试集上预测（构建的测试窗口对应的预测）
    with torch.no_grad():
        test_X = torch.from_numpy(test_dataset.samples.astype(np.float32)).to(device)
        y_pred = model(test_X).cpu().numpy()
        y_true = test_dataset.targets

    metrics = calculate_metrics(y_true, y_pred)
    print('PyTorch LSTM 模型评估指标:')
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 保存预测结果：为了与原脚本保持列名一致，选择将窗口对应的最后时间点与预测匹配
    # 计算测试窗口对应在原 data 中的行索引：测试段起始索引为 train_size
    test_start_idx = train_size
    # 第一个窗口的最后时间步在原测试段中的位置为 test_start_idx + seq_len - 1
    indices = np.arange(test_start_idx + seq_len - 1, test_start_idx + seq_len - 1 + len(y_pred))
    data_test = data.iloc[indices].copy().reset_index(drop=True)
    data_test['PredictedPrice'] = y_pred
    data_test[['DateTime','SpotPrice_RRP','PredictedPrice']].to_csv('pytorch_lstm_predicted_prices.csv', index=False)
    print('预测结果已保存到 pytorch_lstm_predicted_prices.csv')

    # 可视化：绘制预测与实际对比图，并尝试叠加 LightGBM 的预测曲线（若存在）
    try:
        import matplotlib.pyplot as plt
        import os
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 6))
        plt.plot(data_test['DateTime'], data_test['SpotPrice_RRP'], label='实际价格', color='blue')
        plt.plot(data_test['DateTime'], data_test['PredictedPrice'], label='LSTM 预测', color='red', linestyle='--')

        lb_csv = 'lightgbm_predicted_prices.csv'
        if os.path.exists(lb_csv):
            try:
                lb_df = pd.read_csv(lb_csv)
                if 'DateTime' in lb_df.columns:
                    lb_df['DateTime'] = pd.to_datetime(lb_df['DateTime'], errors='coerce')
                # merge on DateTime to align
                merged = pd.merge(data_test[['DateTime']], lb_df[['DateTime','PredictedPrice']], on='DateTime', how='left')
                if 'PredictedPrice' in merged.columns:
                    plt.plot(merged['DateTime'], merged['PredictedPrice'], label='LightGBM 预测', color='green', linestyle=':')
            except Exception as e:
                print('读取 LightGBM 预测文件失败：', e)

        plt.xlabel('时间')
        plt.ylabel('电价')
        plt.title('LSTM 电价预测 vs 实际')
        plt.legend()
        plt.tight_layout()
        plt.savefig('pytorch_lstm_price_comparison.png')
        print('对比图已保存为 pytorch_lstm_price_comparison.png')
        try:
            plt.show()
        except Exception:
            pass
    except Exception:
        print('绘图失败：请安装 matplotlib (pip install matplotlib) 以生成图形')

    # 保存组件
    components = {
        'scaler': scaler,
        'label_encoder': weather_le,
        'features': features,
        'target': target,
        'model_state_dict_path': 'pytorch_lstm_regressor.pth',
        'seq_len': seq_len
    }
    with open('pytorch_lstm_model_components.dill', 'wb') as f:
        dill.dump(components, f)
    print('模型组件已保存到 pytorch_lstm_model_components.dill')


if __name__ == '__main__':
    # 默认使用过去 24 步（例如24小时）作为序列长度，可根据需要修改
    main(seq_len=24)
