import sys

try:
    import pandas as pd
except Exception as e:
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
    # avoid division by zero in MAPE
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

    # IMAPE as in original
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

# ------------- 数据集类 -------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X.astype(np.float32)
        self.y = None if y is None else y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

# ------------- 模型 -------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ------------- 主要流程 -------------

def main():
    # 数据加载
    data = pd.read_csv(r"C:\Users\admin\Desktop\实际数据.csv",
                       skiprows=0,
                       names=['DateTime','SpotPrice_RRP','TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain'])

    # 转换数值列
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

    # 滞后特征
    data['SpotPrice_RRP_lag1'] = pd.to_numeric(data['SpotPrice_RRP'].shift(1), errors='coerce')
    data['TotalDemand_lag1'] = pd.to_numeric(data['TotalDemand'].shift(1), errors='coerce')

    # 丢弃缺失
    data = data.dropna().reset_index(drop=True)

    target = 'SpotPrice_RRP'
    features = ['TotalDemand','WeatherType','Temperature','WindSpeed','Cloud','Humidity','Rain',
                'hour','dayofweek','month','day','SpotPrice_RRP_lag1','TotalDemand_lag1']

    X = data[features].values
    y = data[target].values

    # 划分训练/测试
    train_size = int(len(data) * 0.95)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 保存组件
    components = {
        'scaler': scaler,
        'label_encoder': weather_le,
        'features': features,
        'target': target
    }
    with open('pytorch_model_components.dill', 'wb') as f:
        dill.dump(components, f)

    # Dataset & DataLoader
    train_ds = TimeSeriesDataset(X_train, y_train)
    test_ds = TimeSeriesDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MLPRegressor(input_dim=X_train.shape[1], hidden_dims=[128,64], dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()  # MAE

    # 早停
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

        # 早停保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'pytorch_regressor.pth')
        if epoch - best_epoch >= patience:
            print('Early stopping triggered')
            break

    # 加载最佳模型
    model.load_state_dict(torch.load('pytorch_regressor.pth', map_location=device))
    model.eval()

    # 在测试集上预测
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
        y_pred = model(X_test_t).cpu().numpy()

    # 评估
    metrics = calculate_metrics(y_test, y_pred)
    print('PyTorch 模型评估指标:')
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 保存预测结果
    data_test = data.iloc[train_size:].copy()
    data_test['PredictedPrice'] = y_pred
    data_test[['DateTime','SpotPrice_RRP','PredictedPrice']].to_csv('pytorch_predicted_prices.csv', index=False)
    print('预测结果已保存到 pytorch_predicted_prices.csv')

    # 可视化：绘制预测与实际对比图
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 6))
        plt.plot(data_test['DateTime'], data_test['SpotPrice_RRP'], label='实际价格', color='blue')
        plt.plot(data_test['DateTime'], data_test['PredictedPrice'], label='预测价格', color='red', linestyle='--')
        plt.xlabel('时间')
        plt.ylabel('电价')
        plt.title('PyTorch 电价预测 vs 实际')
        plt.legend()
        plt.tight_layout()
        plt.savefig('pytorch_price_comparison.png')
        print('对比图已保存为 pytorch_price_comparison.png')
        # 在本地运行时可显示
        try:
            plt.show()
        except Exception:
            pass
    except Exception:
        print('绘图依赖缺失：pip install matplotlib 可用来生成对比图')

    # 保存完整组件（包括模型权重路径）
    components['model_state_dict_path'] = 'pytorch_regressor.pth'
    with open('pytorch_model_components.dill', 'wb') as f:
        dill.dump(components, f)
    print('模型组件已保存到 pytorch_model_components.dill')


if __name__ == '__main__':
    main()
