import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(train_losses, val_losses):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

def plot_predictions(actual, predicted, title):
    """绘制预测结果"""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', alpha=0.7)
    plt.plot(predicted, label='Predicted', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.close()

def calculate_metrics(actual, predicted):
    """计算评估指标"""
    # 计算MAPE
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # 计算R²
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MAPE': mape,
        'RMSE': rmse,
        'R2': r2
    }