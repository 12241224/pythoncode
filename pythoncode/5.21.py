import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.axis import DateAxis
import matplotlib.pyplot as plt
import logging
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量
prediction_results = pd.DataFrame()
metrics_results = {}

# =================== 配置参数 ===================
DATA_PATH = r'C:\Users\admin\Desktop\NSW半小时时电价预测数据管理 (1).csv'
PREDICTION_START_DATE = '2024-05-01'
PREDICTION_END_DATE = '2024-05-10'

OUTPUT_EXCEL = '周前预测结果_对比.xlsx'
DATE_FORMAT = '%Y-%m-%d %H:%M'

LAGS = [2, 4]
NUMERIC_COLS = ['Temperature', 'WindSpeed', 'Cloud', 'Humidity', 'Rain']
TARGET_FEATURES = ['SpotPrice_RRP', 'TotalDemand']


# =================== 工具函数 ===================
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    r2 = r2_score(y_true, y_pred)
    value_range = np.max(y_true) - np.min(y_true)
    nmae = mae / value_range if value_range != 0 else np.nan
    nrmse = rmse / value_range if value_range != 0 else np.nan

    pmin, pmax = 30, 350
    imape_values = []

    for true_val, pred_val in zip(y_true, y_pred):
        if true_val < 0:
            if pred_val < 0:
                imape_values.append(0)
            elif 0 <= pred_val <= pmin:
                imape_values.append(0.2)
            else:
                imape_values.append(1)
        elif 0 <= true_val <= pmin:
            if pred_val < 0:
                imape_values.append(0.2)
            elif 0 <= pred_val <= pmin:
                imape_values.append(0)
            else:
                imape_values.append(min(np.abs((pmin - pred_val) / (pmin + 1e-10)), 1))
        elif true_val >= pmax:
            if pred_val < pmax:
                imape_values.append(min(np.abs((pmax - pred_val) / (pmax + 1e-10)), 1))
            else:
                imape_values.append(0)
        else:
            imape_values.append(min(np.abs((true_val - pred_val) / (true_val + 1e-10)), 1))

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


# =================== 数据加载与预处理 ===================
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path, skiprows=1)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y/%m/%d %H:%M')
    df.set_index('DateTime', inplace=True)
    df['WeatherType'] = df['WeatherType'].ffill()
    return df


# =================== 清洗函数：彻底杜绝 NaN ===================
def robust_cleaning(df, numeric_cols):
    """
    强化清洗函数：确保所有数值列无 NaN，并保留 Lag 特征
    """
    df = df.copy()

    logging.info(f"[robust_cleaning] 开始时列：{df.columns.tolist()}")
    df.info()  # 查看初始 DataFrame 状态

    # 确保基础列存在
    required_columns = ['SpotPrice_RRP', 'TotalDemand']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要列 '{col}'，请检查输入数据")

    # 数值列处理
    for col in numeric_cols + required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[numeric_cols + required_columns] = df[numeric_cols + required_columns].ffill().bfill().fillna(0)

    # 天气 one-hot 编码处理
    weather_dummies = pd.get_dummies(df['WeatherType'].astype(str), prefix='weather')
    df = pd.concat([df, weather_dummies], axis=1).fillna(0)

    # 时间特征构建
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # Lag 特征构建前再确认一次目标列是否存在
    assert 'SpotPrice_RRP' in df.columns, "SpotPrice_RRP 列在构建 Lag 前已丢失"
    assert 'TotalDemand' in df.columns, "TotalDemand 列在构建 Lag 前已丢失"

    # Lag 特征构建 + 清洗
    for lag in LAGS:
        df[f'SpotPrice_RRP_lag{lag}'] = df['SpotPrice_RRP'].shift(lag)
        df[f'TotalDemand_lag{lag}'] = df['TotalDemand'].shift(lag)
    lag_cols = [f'{feat}_lag{lag}' for feat in ['SpotPrice_RRP', 'TotalDemand'] for lag in LAGS]
    df[lag_cols] = df[lag_cols].ffill().bfill().fillna(0)

    logging.info(f"[robust_cleaning] 结束后列：{df.columns.tolist()}")
    df.info()  # 再次查看 DataFrame 状态，确保列未丢失

    return df

def build_features(df):
    # 第一步：强化清洗
    df = robust_cleaning(df, NUMERIC_COLS)

    # 检查 lag 列是否存在
    lag_cols = [f'{feat}_lag{lag}' for feat in ['SpotPrice_RRP', 'TotalDemand'] for lag in LAGS]
    missing_cols = [col for col in lag_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下 lag 列缺失，请检查 robust_cleaning(): {missing_cols}")

    # 构建特征列表
    features = ['hour', 'minute', 'dayofweek', 'is_weekend', 'Temperature',
                'WindSpeed', 'Cloud', 'Humidity', 'Rain'] + \
               [col for col in df.columns if col.startswith('weather_')] + \
               [f'SpotPrice_RRP_lag{lag}' for lag in LAGS] + \
               [f'TotalDemand_lag{lag}' for lag in LAGS]

    X = df[features].values
    y = df[TARGET_FEATURES].values

    return df, X, y, features


# =================== 模型训练与预测 ===================
def train_model(X_train, y_train):
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1))
    model.fit(X_train, y_train)
    return model


def predict_future(df, features, model, scaler_X, scaler_y):
    future_dates = pd.date_range(start=f'{PREDICTION_START_DATE} 00:00',
                                 end=f'{PREDICTION_END_DATE} 23:30',
                                 freq='30T')
    future_df = pd.DataFrame(index=future_dates)

    # 时间特征
    future_df['hour'] = future_df.index.hour
    future_df['minute'] = future_df.index.minute
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['is_weekend'] = future_df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # 初始化目标变量
    future_df['SpotPrice_RRP'] = np.nan
    future_df['TotalDemand'] = np.nan

    # 历史均值填充
    last_date = df.index[-1]
    historical_mean = df.loc[last_date - pd.DateOffset(days=7):last_date].mean(numeric_only=True)
    for col in NUMERIC_COLS:
        future_df[col] = historical_mean[col]

    # 天气填充
    weather_cols = [col for col in df.columns if col.startswith('weather_')]
    future_df[weather_cols] = 0
    if df['WeatherType'].notna().any():
        last_weather = df['WeatherType'].dropna().mode()[0]
        future_df[f'weather_{last_weather}'] = 1

    # Lag 特征初始化
    last_4_steps = df[TARGET_FEATURES].iloc[-4:].values
    for i in range(len(future_df)):
        if i < 2:
            future_df.loc[future_df.index[i], 'SpotPrice_RRP_lag2'] = last_4_steps[-2 + i, 0]
            future_df.loc[future_df.index[i], 'TotalDemand_lag2'] = last_4_steps[-2 + i, 1]
            future_df.loc[future_df.index[i], 'SpotPrice_RRP_lag4'] = last_4_steps[-4 + i, 0]
            future_df.loc[future_df.index[i], 'TotalDemand_lag4'] = last_4_steps[-4 + i, 1]
        else:
            future_df.loc[future_df.index[i], 'SpotPrice_RRP_lag2'] = future_df.loc[future_df.index[i - 2], 'SpotPrice_RRP']
            future_df.loc[future_df.index[i], 'TotalDemand_lag2'] = future_df.loc[future_df.index[i - 2], 'TotalDemand']
            future_df.loc[future_df.index[i], 'SpotPrice_RRP_lag4'] = future_df.loc[future_df.index[i - 4], 'SpotPrice_RRP']
            future_df.loc[future_df.index[i], 'TotalDemand_lag4'] = future_df.loc[future_df.index[i - 4], 'TotalDemand']

        if i >= 2:
            current_features = future_df.loc[[future_df.index[i]], features].values
            current_pred = model.predict(scaler_X.transform(current_features))
            unscaled_pred = scaler_y.inverse_transform(current_pred)
            future_df.loc[future_df.index[i], 'SpotPrice_RRP'] = unscaled_pred[0, 0]
            future_df.loc[future_df.index[i], 'TotalDemand'] = unscaled_pred[0, 1]

    # 再次清理预测数据中的 NaN —— 分开处理，防止 SpotPrice_RRP 被影响
    future_df[NUMERIC_COLS] = future_df[NUMERIC_COLS].ffill().bfill().fillna(historical_mean)
    
    # 保留 SpotPrice_RRP 和 TotalDemand，单独处理
    future_df[['SpotPrice_RRP', 'TotalDemand']] = \
        future_df[['SpotPrice_RRP', 'TotalDemand']].ffill().bfill().fillna(method='backfill')

    # 只清洗 feature 相关列，避免污染目标列
    future_df[features] = future_df[features].ffill().bfill().fillna(0)

    global prediction_results
    prediction_results = pd.DataFrame({
        'DateTime': future_df.index,
        'SpotPrice_RRP': future_df['SpotPrice_RRP'],
        'TotalDemand': future_df['TotalDemand']
    })

    return prediction_results


# =================== 结果保存与可视化 ===================
def save_prediction_with_actuals(prediction_df, actual_df, file_path, metrics_data=None):
    try:
        # 合并预测与真实值
        combined = pd.merge(
            prediction_df,
            actual_df[TARGET_FEATURES],
            left_on='DateTime', right_index=True,
            suffixes=('_Pred', '_Actual')
        )

        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 第一个 sheet：预测与实际对比
            combined.to_excel(writer, sheet_name='预测与实际', index=False)
            sheet = writer.sheets['预测与实际']

            sheet.column_dimensions['A'].width = 20
            sheet.column_dimensions['B'].width = 15
            sheet.column_dimensions['C'].width = 15
            sheet.column_dimensions['D'].width = 15
            sheet.column_dimensions['E'].width = 15

            def create_chart(sheet, title, pred_col, act_col, pos):
                chart = LineChart()
                chart.title = title
                chart.y_axis.title = "值"
                chart.x_axis.number_format = 'yyyy-mm-dd hh:mm'
                chart.x_axis.majorTimeUnit = "days"

                pred_ref = Reference(sheet, min_col=pred_col, min_row=1, max_row=len(combined)+1)
                act_ref = Reference(sheet, min_col=act_col, min_row=1, max_row=len(combined)+1)
                date_ref = Reference(sheet, min_col=1, min_row=2, max_row=len(combined)+1)

                chart.add_data(pred_ref, titles_from_data=True)
                chart.add_data(act_ref, titles_from_data=True)
                chart.set_categories(date_ref)
                sheet.add_chart(chart, pos)

            create_chart(sheet, "电价预测 vs 实际", 2, 3, "G2")
            create_chart(sheet, "负荷预测 vs 实际", 4, 5, "G20")

            # 第二个 sheet：评估指标
            if metrics_data is not None:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name='评估指标', index=False)

        logging.info(f"预测与实际已保存至 {file_path}")
        os.startfile(file_path)  # Windows only

    except Exception as e:
        logging.error(f"写入Excel失败: {str(e)}")


# =================== 主流程 ===================
def validate_data(X, y, features):
    X_df = pd.DataFrame(X, columns=features)
    nan_cols = X_df.columns[X_df.isna().any()].tolist()
    if nan_cols:
        raise ValueError(f"以下特征列仍包含 NaN: {nan_cols}")


def main():
    try:
        df = load_and_preprocess_data(DATA_PATH)
        print(df.head())  # 新增打印语句，查看前几行数据
        df, X, y, features = build_features(df)

        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # 训练测试划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=48 * 5 * 2, shuffle=False
        )

        # 检查是否仍有 NaN
        validate_data(X_train, y_train, features)

        # 模型训练
        model = train_model(X_train, y_train)

        # 预测未来数据
        predict_future(df, features, model, scaler_X, scaler_y)

        # 提取真实值
        start_date = pd.to_datetime(PREDICTION_START_DATE)
        end_date = pd.to_datetime(PREDICTION_END_DATE) + pd.Timedelta(hours=23, minutes=30)
        mask_actual = (df.index >= start_date) & (df.index <= end_date)
        actual_df = df.loc[mask_actual].copy()

        # 提取预测值
        mask_pred = (prediction_results['DateTime'] >= start_date) & (prediction_results['DateTime'] <= end_date)
        prediction_df = prediction_results.loc[mask_pred].copy()

        # 合并与评估，并重命名列以避免冲突
        prediction_df_renamed = prediction_df.rename(columns={
            'SpotPrice_RRP': 'SpotPrice_RRP_Pred',
            'TotalDemand': 'TotalDemand_Pred'
        })

        actual_df_restricted = actual_df[TARGET_FEATURES].copy()
        actual_df_restricted = actual_df_restricted.reset_index()

        common_df = pd.merge(
            prediction_df_renamed,
            actual_df_restricted,
            left_on='DateTime', right_on='DateTime'
        )

        print("common_df columns:", common_df.columns.tolist())

        # 计算并保存评估结果
        metrics_data = []
        for feature in TARGET_FEATURES:
            y_true = common_df[feature]
            y_pred = common_df[f'{feature}_Pred']
            metrics = calculate_metrics(y_true, y_pred)
            metrics['Feature'] = feature
            metrics_data.append(metrics)
            metrics_results[feature] = metrics  # 存入全局变量用于导出

        # 打印控制台信息
        print("\n=== 模型评估指标 ===")
        for item in metrics_data:
            print(f"\n【{item['Feature']}】")
            for k, v in item.items():
                if k != 'Feature':
                    print(f"{k}: {v:.4f}")

        # 保存到 Excel
        save_prediction_with_actuals(prediction_df, actual_df, OUTPUT_EXCEL, metrics_data)

    except Exception as e:
        logging.error(f"主程序发生错误: {str(e)}")
        pd.DataFrame({'错误信息': [str(e)]}).to_excel(OUTPUT_EXCEL, sheet_name='错误日志')

if __name__ == '__main__':
    main()