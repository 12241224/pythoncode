import torch
import numpy as np
import pandas as pd
from config import config

def predict_future(model, last_sequence, scaler_X, scaler_y, future_steps):
    """使用模型进行未来预测"""
    model.eval()
    
    # 准备输入数据
    input_seq = last_sequence.copy()
    predictions = []
    
    with torch.no_grad():
        for i in range(future_steps):
            # 标准化输入
            input_reshaped = input_seq.reshape(-1, input_seq.shape[-1])
            input_scaled = scaler_X.transform(input_reshaped).reshape(input_seq.shape)
            
            # 转换为张量
            input_tensor = torch.FloatTensor(input_scaled).to(config.DEVICE)
            
            # 预测下一步
            output = model(input_tensor.transpose(0, 1))
            next_pred = output[-1:].transpose(0, 1).cpu().numpy()
            
            # 反标准化预测结果
            next_pred_reshaped = next_pred.reshape(-1, next_pred.shape[-1])
            next_pred_original = scaler_y.inverse_transform(next_pred_reshaped).reshape(next_pred.shape)
            
            predictions.append(next_pred_original[0])
            
            # 更新输入序列
            # 这里需要根据实际特征结构更新序列
            # 假设最后一个特征是目标值，可以用预测值替换
            new_row = input_seq[-1:].copy()
            # 调试输出 shape
            print("new_row shape:", new_row.shape)
            print("next_pred_original shape:", next_pred_original.shape)
            print("next_pred_original[0] shape:", next_pred_original[0].shape)
            # 只更新目标特征（假设目标特征是前两个）
            price = float(next_pred_original[0][0][0])
            load = float(next_pred_original[0][0][1])
            new_row[0, -1, 0] = price
            new_row[0, -1, 1] = load
            # 更新其他特征（如时间特征等）
            # 这里需要根据实际情况调整
            
            input_seq = np.concatenate([input_seq[1:], new_row], axis=0)
    
    return np.array(predictions)

def save_predictions(predictions, dates, output_path):
    """保存预测结果到Excel"""
    # 创建结果DataFrame
    results = []
    
    for i, (date, pred) in enumerate(zip(dates, predictions)):
        results.append({
            'datetime': date,
            'predicted_price': pred[0][0],
            'predicted_load': pred[0][1]
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_path, index=False)
    
    return results_df