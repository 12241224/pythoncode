import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp

# 页面配置
st.set_page_config(page_title="AI储能决策与竞价系统", layout="wide", page_icon="🔋")

# --- 核心算法：储能优化模型 (对应第三步：竞价策略数学化) ---
def optimize_storage(prices, capacity, max_power, cycle_efficiency):
    """
    使用 PuLP 线性规划计算最优充放电策略
    """
    n_hours = len(prices)
    # 单向转换效率 (假设充放电效率相同，且单向 efficiency = sqrt(循环效率))
    eff = np.sqrt(cycle_efficiency) 
    
    # 定义优化问题 (最大化收益)
    prob = pulp.LpProblem("Battery_Optimization", pulp.LpMaximize)
    
    # 决策变量
    charge = pulp.LpVariable.dicts("Charge", range(n_hours), lowBound=0, upBound=max_power)
    discharge = pulp.LpVariable.dicts("Discharge", range(n_hours), lowBound=0, upBound=max_power)
    soc = pulp.LpVariable.dicts("SOC", range(n_hours), lowBound=0, upBound=capacity)
    
    # 目标函数：放电收益 - 充电成本
    prob += pulp.lpSum(prices[t] * discharge[t] - prices[t] * charge[t] for t in range(n_hours))
    
    # 约束条件
    for t in range(n_hours):
        # SOC 连续性约束 (能量守恒)
        if t == 0:
            prob += soc[t] == 0 + charge[t] * eff - discharge[t] / eff  # 假设期初量为空
        else:
            prob += soc[t] == soc[t-1] + charge[t] * eff - discharge[t] / eff
            
    # 求解，隐藏默认输出日志
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 提取结果
    charge_res = [charge[t].varValue for t in range(n_hours)]
    discharge_res = [discharge[t].varValue for t in range(n_hours)]
    soc_res = [soc[t].varValue for t in range(n_hours)]
    profit = pulp.value(prob.objective)
    
    return charge_res, discharge_res, soc_res, profit

# --- 数据生成函数 (对应第一步：预测模型数据准备) ---
def generate_mock_data(seed=42):
    """模拟生成预测电价和真实电价 (实战场景下这里替换为你基于Darts/lightgbm预测的输出)"""
    np.random.seed(seed)
    hours = list(range(24))
    # 基础双峰电价形态 (早高峰、晚高峰)
    base_price = [0.2, 0.15, 0.1, 0.12, 0.25, 0.5, 0.8, 1.2, 0.9, 0.7, 0.6, 0.5, 
                  0.4, 0.45, 0.6, 0.9, 1.1, 1.3, 1.0, 0.8, 0.6, 0.4, 0.3, 0.25]
    
    # 添加随机噪声作为预测价格
    pred_price = np.maximum(0, np.array(base_price) + np.random.normal(0, 0.1, 24))
    # 真实价格在预测基础上再偏移
    actual_price = np.maximum(0, pred_price + np.random.normal(0, 0.15, 24))
    
    return hours, pred_price.tolist(), actual_price.tolist()

# === UI 前端呈现 (对应第二步和第四步) ===
st.title("🔋 储能单日竞价策略辅助决策平台")
st.markdown("基于 **多维时序预测** 与 **运筹学优化(LP)** 的AI调度决策支持系统")

# 侧边栏：系统参数配置
with st.sidebar:
    st.header("⚙️ 储能资产参数及约束设置")
    capacity = st.number_input("电池铭牌容量 (MWh)", min_value=1.0, max_value=500.0, value=10.0, step=1.0)
    max_power = st.number_input("最大充放电功率 (MW)", min_value=1.0, max_value=100.0, value=5.0, step=1.0)
    cycle_eff = st.slider("储能循环综合效率 (RtE)", min_value=0.70, max_value=1.00, value=0.90, step=0.01)
    
    st.markdown("---")
    st.markdown("### 算法引擎逻辑说明")
    st.info("底层调用求解器基于你的**容量**和**充放电边界**，自动构建 24 个时段的能量流动平衡方程。在满足约束前提下，严格求解出绝对的**理论数学最大化套利收益点位**。")

# 使用 Tabs 分割“明日策略”和“历史回测”
tab1, tab2 = st.tabs(["🚀 T+1 日竞价策略 (模拟未来)", "📊 T-1 日真实盈亏复盘 (历史校验)"])

with tab1:
    st.header("📅 明日储能调度策略与预估盈亏")
    
    # 获取模拟预测数据
    hours, pred_prices, _ = generate_mock_data(seed=100)
    
    # 调用决策优化引擎
    charge_plan, discharge_plan, soc_state, expected_profit = optimize_storage(pred_prices, capacity, max_power, cycle_eff)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("预计明日最大收益", f"¥ {expected_profit:,.2f}", "基于明日预测电价测算")
    col2.metric("规划日充放电量", f"{sum(discharge_plan):.1f} MWh", f"约 {sum(discharge_plan)/capacity:.1f} 次循环")
    col3.metric("度电平均套利净值", f"¥ {expected_profit/sum(discharge_plan) if sum(discharge_plan)>0 else 0:.3f} / kWh")
    
    st.markdown("### 📈 策略联动趋势面版")
    # 绘制带双坐标轴的 Plotly 高阶图表
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. 预测电价线
    fig.add_trace(go.Scatter(x=hours, y=pred_prices, mode='lines+markers', name='模型预测电价 (元/kWh)', line=dict(color='orange', width=2)), secondary_y=False)
    
    # 2. SOC 状态图
    fig.add_trace(go.Scatter(x=hours, y=soc_state, fill='tozeroy', name='电池蓄电状态(SOC) MWh', line=dict(color='LightSkyBlue', width=0), opacity=0.3), secondary_y=True)
    
    # 3. 充放电直方图
    net_power = np.array(discharge_plan) - np.array(charge_plan)
    colors = ['#FF4B4B' if val > 0 else '#00CC96' for val in net_power] # 红放绿充
    fig.add_trace(go.Bar(x=hours, y=net_power, name='充/放电指令 (MW)', marker_color=colors, opacity=0.7), secondary_y=True)
    
    fig.update_layout(height=450, hovermode="x unified", barmode='relative',
                      title_text="预测曲线与调度动作智能叠加分析",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(title_text="自然时间 (小时)", tickmode='linear', tick0=0, dtick=1)
    fig.update_yaxes(title_text="交易价格 (元/kWh)", secondary_y=False)
    fig.update_yaxes(title_text="功率/能量", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 策略表格展示与导出
    plan_df = pd.DataFrame({
        "时段": [f"{h:02d}:00" for h in hours],
        "输入预测电价(元)": [round(p, 4) for p in pred_prices],
        "决策动作": ["放电 ⚡" if d>0 else ("充电 🔋" if c>0 else "待机") for c, d in zip(charge_plan, discharge_plan)],
        "下发功率指令(MW)": [round(d - c, 2) for c, d in zip(charge_plan, discharge_plan)],
        "期末SOC(MWh)": [round(s, 2) for s in soc_state]
    })
    
    with st.expander("📝 展开查看时刻表并导出交易系统"):
        st.dataframe(plan_df, use_container_width=True)
        csv = plan_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出策略文件为 CSV (含UTF-8 BOM)", data=csv, file_name="ai_bidding_plan.csv", mime="text/csv")


with tab2:
    st.header("🔍 真实复盘与夏普比模拟 (盈亏回测校验)")
    st.markdown("逻辑回顾：我们用过去的**预测电价**生成决策，然后再拿该决策去比对现实市场当天发生的**真实电价**计算实际结转盈亏，验证算法抗波动能力。")
    
    past_date = st.date_input("请选择历史回卷日期", pd.to_datetime("today") - pd.Timedelta(days=1))
    
    # 模拟选中日期的回测数据
    h_hours, h_pred_prices, h_actual_prices = generate_mock_data(seed=int(past_date.strftime("%Y%m%d")))
    
    # 第一步：根据当时的【预测电价】拍脑袋做出的决定
    h_charge, h_discharge, _, h_exp_profit = optimize_storage(h_pred_prices, capacity, max_power, cycle_eff)
    
    # 第二步：将这套决定扔进【真实电价】，算出真实结账的钱
    h_actual_profit = sum([h_actual_prices[t]*h_discharge[t] - h_actual_prices[t]*h_charge[t] for t in h_hours])
    
    # 第三步：跑一把“上帝视角”，如果提前知道真实电价，理论能赚的极限情况
    _, _, _, oracle_profit = optimize_storage(h_actual_prices, capacity, max_power, cycle_eff)
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("策略当时预估收益", f"¥ {h_exp_profit:,.2f}", "按预测曲线算的白日梦")
    
    delta_profit = h_actual_profit - h_exp_profit
    col_b.metric("结算实际到手真实收益", f"¥ {h_actual_profit:,.2f}", f"{'+' if delta_profit>0 else ''}{delta_profit:,.2f} 真实波动导致的价差盈亏", delta_color="normal")
    
    oracle_ratio = (h_actual_profit / oracle_profit * 100) if oracle_profit>0 else 0
    col_c.metric("收益达标率 (抗偏差指标)", f"{oracle_ratio:.1f} %", "对比上帝视角的理论极限捕获率")

    st.markdown("### ⚖️ 误差诊断靶场视图")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=h_hours, y=h_pred_prices, mode='lines', line=dict(dash='dash', color='gray'), name='当时的“预测电价”'))
    fig2.add_trace(go.Scatter(x=h_hours, y=h_actual_prices, mode='lines+markers', line=dict(color='purple'), name='当天发生的“真实电价”'))
    
    # 诊断系统：自动圈出因电价预测不准导致的典型“亏钱动作”
    for t in h_hours:
        # 当放电时，实际电价暴跌导致卖便宜了
        if h_discharge[t] > 0 and h_actual_prices[t] < h_pred_prices[t]*0.8:
            fig2.add_annotation(x=t, y=h_actual_prices[t], text="放电踏空", showarrow=True, arrowhead=1, ax=0, ay=-40, font=dict(color="red"))
        # 当充电时，实际电价暴涨导致买贵了    
        if h_charge[t] > 0 and h_actual_prices[t] > h_pred_prices[t]*1.2:
            fig2.add_annotation(x=t, y=h_actual_prices[t], text="高位充电伤", showarrow=True, arrowhead=1, ax=0, ay=40, font=dict(color="green"))
            
    fig2.update_layout(height=400, hovermode="x unified", title_text="预测背离诊断 (标靶标注了被坑的点位)")
    fig2.update_xaxes(title_text="小节 (h)", dtick=1)
    fig2.update_yaxes(title_text="现货电价 (元)")
    st.plotly_chart(fig2, use_container_width=True)
    
    if oracle_ratio < 80:
        st.error("🚨 预警：该日收益达标率不足 80%。这通常说明前端的时序预测模型(LSTM/Darts)预测波峰出现偏移，建议重新检视输入该日的特征工程（如新能源风力冲击、天气突变未捕捉等）。")
    else:
        st.success("✅ 诊断：当日预测基本拟合了核心的高低价波段缝隙，算法抗波动作战成功。")