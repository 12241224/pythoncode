import pandas as pd
import numpy as np
from config import config

def load_data():
    """加载电价数据"""
    df = pd.read_csv(config.DATA_PATH)
    
    # 确保时间格式正确
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)

    return df

def load_actual_data():
    """加载实际值数据"""
    df = pd.read_csv(config.ACTUAL_DATA_PATH)
    
    # 确保时间格式正确
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    return df

def split_data(df):
    """划分训练集和测试集"""
    # 计算分割点
    split_idx = int(len(df) * config.TRAIN_RATIO)
    
    train_data = df.iloc[:split_idx]
    test_data = df.iloc[split_idx:]
    
    return train_data, test_data