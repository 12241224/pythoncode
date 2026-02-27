import pandas as pd
import numpy as np
from config import config

def create_features(df, is_training=True):
    print("DataFrame columns:", df.columns)
    """创建时间特征和天气特征"""
    df = df.copy()

    # 提取时间特征
    df['hour'] = df['DateTime'].dt.hour + df['DateTime'].dt.minute / 60
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['is_weekend'] = df['DateTime'].dt.dayofweek >= 5

    # 如果是预测阶段，使用晴天假设
    if not is_training:
        df['cloud'] = 0  # 假设无云
        df['rain'] = 0   # 假设无雨

    # 选择需要的特征
    all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES + config.WEATHER_FEATURES
    df = df[all_features]

    return df

def create_sequences(data, target_cols=['SpotPrice_RRP', 'TotalDemand']):
    """创建时间序列样本"""
    X, y = [], []
    
    for i in range(len(data) - config.PAST_STEPS - config.FUTURE_STEPS + 1):
        # 过去时间步的特征
        past_features = data.iloc[i:i+config.PAST_STEPS]
        
        # 未来时间步的目标值
        future_targets = data[target_cols].iloc[i+config.PAST_STEPS:i+config.PAST_STEPS+config.FUTURE_STEPS]
        
        X.append(past_features)
        y.append(future_targets.values)
    
    return np.array(X), np.array(y)