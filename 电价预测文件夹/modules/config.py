import torch

class Config:
    # 数据路径
    DATA_PATH = r"C:\Users\admin\Desktop\杂物文件夹\NSW半小时时电价预测数据管理 (1).csv"
    OUTPUT_PATH = "forecast_results.xlsx"
    ACTUAL_DATA_PATH = r"C:\Users\admin\Desktop\实际数据.csv"
    
    # 时间参数
    PAST_STEPS = 48  # 使用过去4天的数据(4*24=96个时间点)
    FUTURE_STEPS = 48  # 预测未来一天(48个时间点，每半小时一个)

    # 模型参数
    D_MODEL = 32  # Transformer的嵌入维度
    NHEAD = 2  # 注意力头数
    NUM_LAYERS = 2  # Transformer层数
    DROPOUT = 0.1  # Dropout率
    
    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    TRAIN_RATIO = 0.8  # 训练集比例
    
    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 特征配置
    NUMERICAL_FEATURES = [
        'SpotPrice_RRP', 'TotalDemand', 'Temperature', 'Humidity', 'WindSpeed'
    ]
    CATEGORICAL_FEATURES = [
        'hour', 'day_of_week', 'month', 'is_weekend'
    ]
    # 天气特征 - 如果使用晴天假设，这些特征可以设为默认值
    WEATHER_FEATURES = ['Cloud', 'Rain']
    
config = Config()