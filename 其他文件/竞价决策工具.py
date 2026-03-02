"""
储能商家竞价决策工具
基于预测价格，提供实时最优竞价建议
"""

import pandas as pd
import numpy as np

class StorageArbiBiddingAdvisor:
    """储能套利竞价顾问"""
    
    def __init__(self, predictions: np.ndarray, 
                 capacity_mw: float = 100,
                 efficiency: float = 0.85,
                 variable_cost: float = 50):
        """
        参数:
            predictions: 24小时电价预测 (元/MWh)
            capacity_mw: 储能容量 (MW)
            efficiency: 充放电效率 (%)
            variable_cost: 可变成本 (元/MWh)
        """
        self.pred_prices = predictions
        self.capacity = capacity_mw
        self.efficiency = efficiency
        self.var_cost = variable_cost
        
        # 分峰谷定义
        self.valley_hours = [0, 1, 2, 3, 4, 5]
        self.morning_peak = [8, 9, 10, 11]
        self.evening_peak = [18, 19, 20, 21]
        
    def calculate_bidding_price(self) -> dict:
        """计算各时段最优竞价"""
        
        advice = {}
        mean_price = np.mean(self.pred_prices)
        
        # 放电竞价 (出卖电量)
        discharge_advice = {}
        for hour in range(24):
            price = self.pred_prices[hour]
            
            # 基础逻辑：可变成本是底线
            min_bid = self.var_cost + 5  # 5元/MWh最小利润
            
            if hour in self.morning_peak:
                # 早高峰：积极竞价
                bid_price = price - 3  # 略低于预测，保证中标
                confidence = 'high'
                reason = '早高峰需求高，易中标'
                
            elif hour in self.evening_peak:
                # 晚高峰：保守竞价
                bid_price = price - 2
                confidence = 'medium'
                reason = '晚高峰稳定价格，正常竞价'
                
            elif price > mean_price * 1.15:
                # 其他高价时段
                bid_price = price - 4
                confidence = 'high'
                reason = '较高价格时段，应积极参与'
                
            elif price > mean_price:
                # 中等价格
                bid_price = price - 1
                confidence = 'medium'
                reason = '中等价格，保守参与'
                
            else:
                # 低价时段不放电
                bid_price = None
                confidence = 'none'
                reason = '低价时段，不建议放电'
            
            if bid_price is not None and bid_price >= min_bid:
                profit_per_mwh = bid_price - self.var_cost
                discharge_advice[hour] = {
                    'bid_price': f'{bid_price:.2f}',
                    'pred_price': f'{price:.2f}',
                    'profit_margin': f'{profit_per_mwh:.2f}',
                    'daily_profit': f'￥{profit_per_mwh * self.capacity:.0f}',
                    'confidence': confidence,
                    'reason': reason
                }
        
        # 充电竞价 (购买电量)
        charge_advice = {}
        for hour in range(24):
            price = self.pred_prices[hour]
            
            # 最高可接受充电价
            max_charge_price = mean_price * 0.6  # 平均价60%以下才充
            
            if price <= max_charge_price:
                # 积极充电
                bid_price = price + 1  # 略高于预测确保购到
                confidence = 'high'
                reason = '低价时段，优先充电'
                
            elif price <= mean_price * 0.7:
                # 可以充电
                bid_price = price + 0.5
                confidence = 'medium'
                reason = '较低价格，可考虑充电'
                
            else:
                # 不充电
                bid_price = None
                confidence = 'none'
                reason = '价格不低，不推荐充电'
            
            if bid_price is not None:
                charge_advice[hour] = {
                    'bid_price': f'{bid_price:.2f}',
                    'pred_price': f'{price:.2f}',
                    'charging_cost': f'￥{bid_price * self.capacity:.0f}',
                    'confidence': confidence,
                    'reason': reason
                }
        
        return {
            'discharge': discharge_advice,
            'charge': charge_advice,
            'mean_price': mean_price,
            'price_std': np.std(self.pred_prices)
        }
    
    def get_optimal_arbitrage(self) -> dict:
        """获取最优套利方案"""
        
        # 找到最低充电价和最高放电价
        min_price_hour = np.argmin(self.pred_prices)
        max_price_hour = np.argmax(self.pred_prices)
        
        min_price = self.pred_prices[min_price_hour]
        max_price = self.pred_prices[max_price_hour]
        
        price_spread = max_price - min_price
        
        # 检查时间顺序是否合理（放电在充电之后）
        if max_price_hour > min_price_hour:
            time_ok = True
            time_gap = max_price_hour - min_price_hour
        else:
            # 找第二高价
            sorted_prices = np.argsort(self.pred_prices)[::-1]
            for high_hour in sorted_prices:
                if high_hour > min_price_hour:
                    max_price_hour = high_hour
                    max_price = self.pred_prices[high_hour]
                    price_spread = max_price - min_price
                    time_ok = True
                    time_gap = max_price_hour - min_price_hour
                    break
            else:
                time_ok = False
                time_gap = 0
        
        # 计算利润
        if time_ok:
            gross_profit = price_spread * self.capacity
            net_profit = gross_profit * self.efficiency
            charging_cost = min_price * self.capacity
            revenue = max_price * self.capacity * self.efficiency
        else:
            net_profit = 0
            charging_cost = 0
            revenue = 0
        
        return {
            'best_charge_hour': int(min_price_hour),
            'best_discharge_hour': int(max_price_hour),
            'charge_price': f'{min_price:.2f}',
            'discharge_price': f'{max_price:.2f}',
            'price_spread': f'{price_spread:.2f}',
            'time_gap_hours': int(time_gap),
            'feasible': time_ok,
            'charging_cost': f'￥{charging_cost:.0f}',
            'revenue': f'￥{revenue:.0f}',
            'net_profit': f'￥{net_profit:.0f}',
            'profit_per_cycle': f'￥{net_profit * self.efficiency:.0f}',
            'roi_single_cycle': f'{(net_profit / (charging_cost + 1)) * 100:.1f}%'
        }
    
    def generate_daily_strategy(self) -> str:
        """生成每日竞价策略报告"""
        
        bidding_advice = self.calculate_bidding_price()
        arbitrage = self.get_optimal_arbitrage()
        
        report = f"""
{'='*80}
             储能商家每日竞价最优策略报告
{'='*80}

【1】市场环境分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
平均电价: {bidding_advice['mean_price']:.2f} 元/MWh
价格波动: σ = {bidding_advice['price_std']:.2f} 元/MWh
预测范围: {self.pred_prices.min():.2f} - {self.pred_prices.max():.2f} 元/MWh
价格系数(CV): {bidding_advice['price_std']/bidding_advice['mean_price']*100:.1f}%

市场性质: {'价格波动大，套利机会好' if bidding_advice['price_std'] > 15 else '价格相对稳定，套利机会有限'}


【2】最优套利指标
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
最佳充电时段: {arbitrage['best_charge_hour']:02d}:00 @ {arbitrage['charge_price']} 元/MWh
最佳放电时段: {arbitrage['best_discharge_hour']:02d}:00 @ {arbitrage['discharge_price']} 元/MWh
时间间隔: {arbitrage['time_gap_hours']} 小时

价差: {arbitrage['price_spread']} 元/MWh
充电成本: {arbitrage['charging_cost']}
预期收入: {arbitrage['revenue']}
单次净利润: {arbitrage['net_profit']}
单次ROI: {arbitrage['roi_single_cycle']}

可行性: {'✓ 时间顺序合理' if arbitrage['feasible'] else '✗ 需调整策略'}


【3】实时放电竞价建议（出卖电量）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for hour, advice in bidding_advice['discharge'].items():
            report += f"""
时段 {hour:02d}:00 {'(晚高峰)' if hour in self.evening_peak else '(早高峰)' if hour in self.morning_peak else '(其他)'}
  ├─ 预测价格: {advice['pred_price']} 元/MWh
  ├─ 建议竞价: {advice['bid_price']} 元/MWh (自信度: {advice['confidence']})
  ├─ 利润空间: {advice['profit_margin']} 元/MWh
  ├─ 预期日利: {advice['daily_profit']}
  └─ 原因: {advice['reason']}"""
        
        if not bidding_advice['discharge']:
            report += "\n  (当前预测中无合适的放电时段)"
        
        report += f"""

【4】实时充电竞价建议（购买电量）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for hour, advice in bidding_advice['charge'].items():
            report += f"""
时段 {hour:02d}:00 {'(谷底)' if hour in self.valley_hours else '(其他)'}
  ├─ 预测价格: {advice['pred_price']} 元/MWh
  ├─ 建议竞价: {advice['bid_price']} 元/MWh (自信度: {advice['confidence']})
  ├─ 充电成本: {advice['charging_cost']}
  └─ 原因: {advice['reason']}"""
        
        if not bidding_advice['charge']:
            report += "\n  (当前预测中无合适的充电时段)"
        
        report += f"""

【5】风险提示
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 预测不确定性: ±19.78 元/MWh (95%置信区间)
⚠️ 极端价格可能性: 存在
⚠️ 网络拥塞: 可能影响中标概率
⚠️ 政策风险: 实时电价政策可能调整
⚠️ 技术风险: 充放电故障导致无法交付

建议对冲措施:
• 竞价时留5-10%的安全边际
• 不在极端时段追求最高/最低价
• 保留20%容量作为应急备用
• 实时监测市场价格，灵活调整

{'='*80}
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""
        
        return report


# ============ 实例使用 ============
if __name__ == '__main__':
    # 读取预测结果
    results = pd.read_csv('d:\\pythoncode\\储能电价预测结果.csv')
    predictions = results['predicted_price'].values
    
    # 创建竞价顾问
    advisor = StorageArbiBiddingAdvisor(
        predictions=predictions,
        capacity_mw=100,
        efficiency=0.85,
        variable_cost=45
    )
    
    # 生成策略报告
    strategy = advisor.generate_daily_strategy()
    print(strategy)
    
    # 保存报告
    with open('d:\\pythoncode\\每日竞价策略.txt', 'w', encoding='utf-8') as f:
        f.write(strategy)
    
    print("\n✓ 策略报告已保存至: 每日竞价策略.txt")
