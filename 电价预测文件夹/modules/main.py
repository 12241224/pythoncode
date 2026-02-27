
import warnings
warnings.filterwarnings('ignore')

import traceback
import numpy as np
import pandas as pd
from data_loader import load_data, load_actual_data, split_data
from feature_engineer import create_features, create_sequences
from train import train_model
from predict import predict_future, save_predictions
from utils import plot_training_history, plot_predictions, calculate_metrics
from config import config

def main():
    """主函数"""
    print("开始电价预测项目...")
    
    try:
        # 1. 加载数据
        print("加载数据...")
        df = load_data()
        print(f"数据形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")
        
        # 2. 特征工程
        print("进行特征工程...")
        df_featured = create_features(df, is_training=True)
        print(f"特征工程后数据形状: {df_featured.shape}")
        print(f"特征工程后列名: {list(df_featured.columns)}")
        
        # 3. 创建序列数据
        print("创建时间序列...")
        X, y = create_sequences(df_featured)
        print(f"X形状: {X.shape}, y形状: {y.shape}")
        
        # 检查是否有足够的样本
        if X.shape[0] == 0:
            print("错误: 没有创建任何序列样本!")
            print(f"数据长度: {len(df_featured)}")
            print(f"PAST_STEPS: {config.PAST_STEPS}")
            print(f"FUTURE_STEPS: {config.FUTURE_STEPS}")
            return
        
        # 4. 划分训练集和验证集
        split_idx = int(X.shape[0] * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"验证集形状: X_val={X_val.shape}, y_val={y_val.shape}")
        
        # 5. 训练模型
        print("训练模型...")
        model, scaler_X, scaler_y, train_losses, val_losses = train_model(X_train, y_train, X_val, y_val)
        
        # 6. 绘制训练历史
        plot_training_history(train_losses, val_losses)
        
        # 7. 准备预测数据
        print("准备预测数据...")
        # 使用最后一段数据作为预测起点
        last_sequence = X[-1:]
        print(f"最后序列形状: {last_sequence.shape}")
        
        # 8. 进行预测
        print("进行预测...")
        predictions = predict_future(model, last_sequence, scaler_X, scaler_y, config.FUTURE_STEPS)
        print(f"预测结果形状: {predictions.shape}")
        
        # 9. 生成预测日期
        last_date = df['DateTime'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(minutes=30),
            periods=config.FUTURE_STEPS,
            freq='30min'
        )
        print(f"生成预测日期: {len(future_dates)}个时间点")
        
        # 10. 保存结果
        print("保存预测结果...")
        results_df = save_predictions(predictions, future_dates, config.OUTPUT_PATH)
        print(f"结果已保存到 {config.OUTPUT_PATH}")
        
        # 11. 加载实际值数据并进行比较
        print("加载实际值数据...")
        try:
            actual_df = load_actual_data()
            
            # 确保实际值数据包含预测时间段
            actual_values = actual_df[actual_df['DateTime'].isin(future_dates)]
            
            if len(actual_values) > 0:
                # 按时间排序
                actual_values = actual_values.sort_values('DateTime')
                
                # 确保实际值和预测值的时间对齐
                aligned_predictions = []
                for date in future_dates:
                    if date in actual_values['DateTime'].values:
                        idx = actual_values[actual_values['DateTime'] == date].index[0]
                        aligned_predictions.append(predictions[list(future_dates).index(date)])
                    else:
                        aligned_predictions.append([np.nan, np.nan])  # 如果没有实际值，使用NaN
                
                aligned_predictions = np.array(aligned_predictions)
                aligned_predictions = aligned_predictions.reshape(-1, 2)
                
                # 提取实际价格和负荷
                actual_prices = actual_values['SpotPrice_RRP'].values
                actual_loads = actual_values['TotalDemand'].values
                
                # 计算评估指标
                # 只计算有实际值的点
                valid_indices = ~np.isnan(actual_prices) & ~np.isnan(aligned_predictions[:, 0])
                
                if np.sum(valid_indices) > 0:
                    price_metrics = calculate_metrics(
                        actual_prices[valid_indices], 
                        aligned_predictions[valid_indices, 0]
                    )
                    load_metrics = calculate_metrics(
                        actual_loads[valid_indices], 
                        aligned_predictions[valid_indices, 1]
                    )
                    
                    print("电价预测指标:")
                    for metric, value in price_metrics.items():
                        print(f"{metric}: {value:.4f}")
                    
                    print("\n负荷预测指标:")
                    for metric, value in load_metrics.items():
                        print(f"{metric}: {value:.4f}")
                    
                    # 绘制预测结果
                    plot_predictions(
                        actual_prices[:min(48*7, len(actual_prices))],  # 第一周的实际电价
                        aligned_predictions[:min(48*7, len(aligned_predictions)), 0],  # 第一周的预测电价
                        "Price Prediction - First Week"
                    )
                    
                    # 保存比较结果
                    comparison_df = pd.DataFrame({
                        'DateTime': future_dates,
                        'predicted_price': aligned_predictions[:, 0],
                        'actual_price': actual_prices,
                        'predicted_load': aligned_predictions[:, 1],
                        'actual_load': actual_loads
                    })
                    
                    comparison_df.to_excel('comparison_results.xlsx', index=False)
                    print("比较结果已保存到 comparison_results.xlsx")
                else:
                    print("没有有效的数据点可用于比较")
            else:
                print("实际值数据中没有找到预测时间段的数据")
        except FileNotFoundError:
            print("实际值文件未找到，跳过比较步骤")
        except Exception as e:
            print(f"加载实际值数据时出错: {str(e)}")
        
        print("预测完成！")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()