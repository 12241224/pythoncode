import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
from openpyxl.chart import LineChart, Reference
import os
import logging
import traceback
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UC_Model:
    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
        self.create_model()
    
    def get_unit_param_as_float(self, unit_dict, unit, param_name, default=0.0):
        """改进的参数获取方法，处理各种数据类型和缺失值"""
        try:
            if unit not in unit_dict:
                logging.warning(f"机组 {unit} 不在字典中，使用默认值 {default}")
                return default
                
            unit_data = unit_dict[unit]
            
            # 定义列名映射
            param_name_mapping = {
                'Capacity': ['Capacity', '机组装机容量(MW)', '装机容量', '容量', 'MaxPower'],
                'MinPower': ['MinPower', 'Minpower', '最小出力', '最小技术出力'],
                'OperationCoeff_A': ['OperationCoeff_A', 'Coeff_A', 'A系数'],
                'OperationCoeff_B': ['OperationCoeff_B', 'Coeff_B', 'B系数'],
                'OperationCoeff_C': ['OperationCoeff_C', 'Coeff_C', 'C系数'],
                'UpRamppingRate': ['UpRamppingRate', 'UpRampingRate', '爬坡速率', '上爬坡速率'],
                'DownRamppingRate': ['DownRamppingRate', 'DownRampingRate', '下爬坡速率'],
                'MinOnLineTime': ['MinOnLineTime', 'MinOnTime', '最小运行时间'],
                'MinOffLineTime': ['MinOffLineTime', 'MinOffTime', '最小停机时间'],
                'DeepPeakShavingThreshold': ['DeepPeakShavingThreshold', '深度调峰阈值'],
                'DeepPeakShavingCost': ['DeepPeakShavingCost', '深度调峰成本'],
                'StateOnFuelConsumption': ['StateOnFuelConsumption', '启动燃料消耗', '启停成本'],
                'InitStatus': ['InitStatus', '初始状态'],
                'ChargeEfficiency': ['ChargeEfficiency', '充电效率'],
                'DisEfficiency': ['DisEfficiency', '放电效率'],
                'MaxChargePower': ['MaxChargePower', '最大充电功率'],
                'MaxDisPower': ['MaxDisPower', 'MaxDischargePower', '最大放电功率'],
                'MinSOC': ['MinSOC', '最小SOC'],
                'MaxSOC': ['MaxSOC', '最大SOC'],
                'InitSOC': ['InitSOC', '初始SOC']
            }
            
            # 获取可能的列名列表
            possible_names = param_name_mapping.get(param_name, [param_name])
            
            # 查找实际存在的列名
            actual_param_name = None
            for name in possible_names:
                if name in unit_data:
                    actual_param_name = name
                    break
            
            if actual_param_name:
                value = unit_data[actual_param_name]
                
                # 处理各种数据类型
                if pd.isna(value):
                    return default
                elif isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # 处理布尔值字符串
                    if value.strip().lower() in ['true', 'yes', '1']:
                        return 1.0
                    elif value.strip().lower() in ['false', 'no', '0']:
                        return 0.0
                    # 处理科学计数法
                    elif 'e' in value.lower():
                        return float(value)
                    # 处理百分数
                    elif '%' in value:
                        return float(value.strip('%')) / 100.0
                    # 尝试直接转换
                    try:
                        return float(value)
                    except:
                        return default
                else:
                    return default
            else:
                return default
        except Exception as e:
            logging.warning(f"获取参数 {param_name} 失败: {str(e)}")
            return default

    def get_unit_param_as_int(self, unit_dict, unit, param_name, default=0):
        """获取整数参数"""
        float_value = self.get_unit_param_as_float(unit_dict, unit, param_name, float(default))
        return int(round(float_value))

    def load_data(self):
        """加载所有Excel数据"""
        logging.info("开始加载数据...")
        
        try:
            # 系统汇总信息
            self.system_summary = pd.read_excel(
                self.data_path, sheet_name='SystemSummary', skiprows=0, nrows=1
            ).iloc[0]
            
            # 机组参数 - 跳过标题行，直接读取数据
            self.thermal_units = pd.read_excel(
                self.data_path, sheet_name='UnitThermalGenerators', skiprows=1
            ).dropna(subset=['ThermalUnitNumber'])
            
            self.wind_units = pd.read_excel(
                self.data_path, sheet_name='UnitWindGenerators', skiprows=1
            ).dropna(subset=['WTPlantID'])
            
            self.solar_units = pd.read_excel(
                self.data_path, sheet_name='UnitSolarGenerators', skiprows=1
            ).dropna(subset=['PVPlantID'])
            
            self.ess_units = pd.read_excel(
                self.data_path, sheet_name='StorElectrochemicalESS', skiprows=1
            ).dropna(subset=['ESSId'])
            
            self.hydro_units = pd.read_excel(
                self.data_path, sheet_name='UnitRunoffHydroGenerators', skiprows=1
            ).dropna(subset=['RunoffHydroUnitName'])
            
            # 曲线数据
            self.wind_curves = pd.read_excel(
                self.data_path, sheet_name='CurveWindResource', skiprows=1
            ).dropna(subset=['WTCurve'])
            
            self.solar_curves = pd.read_excel(
                self.data_path, sheet_name='CurveSolarResource', skiprows=1
            ).dropna(subset=['PVCurve'])
            
            self.hydro_curves = pd.read_excel(
                self.data_path, sheet_name='CurveRunoffHydroResource', skiprows=1
            ).dropna(subset=['RunoffHydroOperationCurve'])
            
            # 处理负荷曲线
            load_curve_df = pd.read_excel(
                self.data_path, sheet_name='CurveLoad', skiprows=1
            )
            
            # 重命名时间列为统一格式
            time_columns = [col for col in load_curve_df.columns if col != 'LoadCurve']
            for i, col in enumerate(time_columns):
                load_curve_df.rename(columns={col: f'Time_{i}'}, inplace=True)
            
            if not load_curve_df.empty:
                self.load_curve = load_curve_df.iloc[0]
                # 确保所有时间点的值都是浮点数
                for col in self.load_curve.index:
                    if col.startswith('Time_'):
                        try:
                            self.load_curve[col] = float(self.load_curve[col])
                        except (ValueError, TypeError):
                            self.load_curve[col] = 0.0
            else:
                # 创建默认负荷曲线
                self.load_curve = pd.Series([0.8] * 24, index=[f'Time_{i}' for i in range(24)])
                logging.warning("负荷曲线数据为空，使用默认值")
            
            # 网络数据
            self.buses = pd.read_excel(self.data_path, sheet_name='NetBuses', skiprows=1).dropna(subset=['BusId'])
            self.lines = pd.read_excel(self.data_path, sheet_name='NetLines', skiprows=1).dropna(subset=['BranchId'])
            self.transformers = pd.read_excel(self.data_path, sheet_name='NetTransformers', skiprows=1).dropna(subset=['TransformerId'])
            self.sections = pd.read_excel(self.data_path, sheet_name='NetSectionsLines', skiprows=1).dropna(subset=['NetSectionId', 'BranchId'])
            self.section_capacity = pd.read_excel(self.data_path, sheet_name='NetSectionsCapacity', skiprows=1).dropna(subset=['NetSectionId'])
            self.ac_tielines = pd.read_excel(self.data_path, sheet_name='NetHVACTieLine', skiprows=1).dropna(subset=['ACTieLineId'])
            self.ac_tieline_curves = pd.read_excel(self.data_path, sheet_name='EnergyHVACTieLine', skiprows=0).dropna(subset=['资源曲线名称'])
            
            # 负荷数据
            self.loads = pd.read_excel(self.data_path, sheet_name='Loads', skiprows=0)
            # 重命名列以解决中文列名问题
            self.loads = self.loads.rename(columns={
                '负荷编号': 'LoadId',
                '负荷名称': 'LoadName',
                '所在母线编号': 'BusName',
                '所属区域名称': 'AreaName',
                '负荷曲线名称': 'LodCurveName',
                '有功比例系数': 'ActivePowerCoef',
                '最小响应能力': 'MinResponse',
                '最大响应能力': 'MaxResponse',
                '是否可中断': 'IsInterrupt',
                '是否可转移': 'IsTransforable',
                '中断成本（元/MW）': 'InterruptCost',
                '转移成本（元/MW）': 'TransforCost',
                '最大响应持续时间(h)': 'MaxResponseDuration',
                '最小响应持续时间(h)': 'MinResponseDuration',
                '最大响应次数': 'MaxResponseTimes',
                '单位补贴（元/MW）': 'Subsidy',
                '可响应时段': 'ResponsePeriod',
                '报量1(为最大最小响应能力之间的比例)': 'BidQuantity1',
                '报价1(元/MWh)': 'BidPrice1',
                '报量2': 'BidQuantity2',
                '报价2(元/MWh)': 'BidPrice2',
                '报量3': 'BidQuantity3',
                '报价3(元/MWh)': 'BidPrice3'
            })
            
            # 确保有功比例系数是数值类型
            self.loads['ActivePowerCoef'] = pd.to_numeric(self.loads['ActivePowerCoef'], errors='coerce').fillna(0)
            
            # 设置时间参数
            self.T = int(self.system_summary['Duration'])
            self.dt = 1  # 调度间隔为1小时
            self.hours = list(range(self.T))
            
            # 创建机组字典
            self.thermal_dict = self.thermal_units.set_index('ThermalUnitNumber').to_dict('index') if not self.thermal_units.empty else {}
            self.wind_dict = self.wind_units.set_index('WTPlantID').to_dict('index') if not self.wind_units.empty else {}
            self.solar_dict = self.solar_units.set_index('PVPlantID').to_dict('index') if not self.solar_units.empty else {}
            self.ess_dict = self.ess_units.set_index('ESSId').to_dict('index') if not self.ess_units.empty else {}
            self.hydro_dict = self.hydro_units.set_index('RunoffHydroUnitName').to_dict('index') if not self.hydro_units.empty else {}
            
            logging.info("数据加载完成")
            return True
        except Exception as e:
            logging.error(f"加载数据失败: {str(e)}")
            traceback.print_exc()
            return False

    def create_model(self):
        """创建优化模型 - 简化版，只包含核心约束"""
        logging.info("开始创建优化模型...")
        
        try:
            self.model = pyo.ConcreteModel()
            
            # 定义集合
            self.model.T = pyo.Set(initialize=self.hours)  # 时间段
            self.model.ThermalUnits = pyo.Set(initialize=self.thermal_units['ThermalUnitNumber'].tolist())
            self.model.WindUnits = pyo.Set(initialize=self.wind_units['WTPlantID'].tolist())
            self.model.SolarUnits = pyo.Set(initialize=self.solar_units['PVPlantID'].tolist())
            self.model.ESSUntis = pyo.Set(initialize=self.ess_units['ESSId'].tolist())
            self.model.HydroUnits = pyo.Set(initialize=self.hydro_units['RunoffHydroUnitName'].tolist())
            self.model.Loads = pyo.Set(initialize=self.loads['LoadId'].tolist())
            
            # 定义变量
            # 火电机组
            self.model.thermal_status = pyo.Var(
                self.model.ThermalUnits, self.model.T, within=pyo.Binary
            )
            self.model.thermal_power = pyo.Var(
                self.model.ThermalUnits, self.model.T, within=pyo.NonNegativeReals
            )
            
            # 可再生能源
            self.model.wind_power = pyo.Var(
                self.model.WindUnits, self.model.T, within=pyo.NonNegativeReals
            )
            self.model.solar_power = pyo.Var(
                self.model.SolarUnits, self.model.T, within=pyo.NonNegativeReals
            )
            
            # 目标函数 - 简化的线性成本函数
            def total_cost_rule(model):
                cost = 0
                # 火电运行成本 (线性近似)
                for unit in model.ThermalUnits:
                    for t in model.T:
                        b = self.get_unit_param_as_float(self.thermal_dict, unit, 'OperationCoeff_B', 0.0)
                        c = self.get_unit_param_as_float(self.thermal_dict, unit, 'OperationCoeff_C', 0.0)
                        cost += b * model.thermal_power[unit, t] + c * model.thermal_status[unit, t]
                return cost
            
            self.model.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)
            
            # 约束条件
            # 1. 火电机组出力约束
            def thermal_power_rule(model, unit, t):
                min_power = self.get_unit_param_as_float(self.thermal_dict, unit, 'MinPower', 0.0)
                max_power = self.get_unit_param_as_float(self.thermal_dict, unit, 'Capacity', 0.0)
                return (min_power * model.thermal_status[unit, t], 
                        model.thermal_power[unit, t], 
                        max_power * model.thermal_status[unit, t])
            
            self.model.thermal_power_con = pyo.Constraint(
                self.model.ThermalUnits, self.model.T, rule=thermal_power_rule
            )
            
            # 2. 可再生能源出力约束
            def wind_power_rule(model, unit, t):
                curve_name = self.wind_dict[unit]['ResourceCurve']
                capacity = self.get_unit_param_as_float(self.wind_dict, unit, 'Capacity', 0.0)
                
                # 从风电曲线中获取预测值
                wind_curve = self.wind_curves[self.wind_curves['WTCurve'] == curve_name].iloc[0]
                time_col = f'Time_{t}'
                if time_col in wind_curve:
                    max_power = wind_curve[time_col] * capacity
                else:
                    max_power = 0
                return model.wind_power[unit, t] <= max_power
            
            self.model.wind_power_con = pyo.Constraint(
                self.model.WindUnits, self.model.T, rule=wind_power_rule
            )
            
            # 3. 系统功率平衡约束
            def power_balance_rule(model, t):
                # 总发电
                thermal_gen = sum(model.thermal_power[unit, t] for unit in model.ThermalUnits)
                wind_gen = sum(model.wind_power[unit, t] for unit in model.WindUnits)
                solar_gen = sum(model.solar_power[unit, t] for unit in model.SolarUnits)
                
                # 总负荷
                total_load = 0
                for load in model.Loads:
                    # 获取该负荷的有功比例系数
                    active_coef = self.loads.loc[self.loads['LoadId'] == load, 'ActivePowerCoef'].values[0]
                    # 获取负荷曲线值
                    time_key = f'Time_{t}'
                    if time_key in self.load_curve:
                        load_curve_val = self.load_curve[time_key]
                    else:
                        load_curve_val = 0.0
                    total_load += active_coef * load_curve_val
                
                return thermal_gen + wind_gen + solar_gen == total_load
            
            self.model.power_balance_con = pyo.Constraint(
                self.model.T, rule=power_balance_rule
            )
            
            logging.info("优化模型创建完成")
            return True
        except Exception as e:
            logging.error(f"创建模型失败: {str(e)}")
            traceback.print_exc()
            return False

    def solve(self):
        """求解优化问题"""
        logging.info("开始求解优化问题...")
        
        try:
            # 使用GLPK求解器 (免费)
            solver = SolverFactory('glpk')
            results = solver.solve(self.model, tee=True)
            
            if (results.solver.status == pyo.SolverStatus.ok and 
                results.solver.termination_condition == pyo.TerminationCondition.optimal):
                logging.info("优化求解成功完成")
                return True
            else:
                logging.error(f"优化求解失败: {results.solver.termination_condition}")
                return False
        except Exception as e:
            logging.error(f"求解失败: {str(e)}")
            return False

    def save_results(self, output_path):
        """保存结果到Excel"""
        logging.info("开始保存结果到Excel...")
        
        try:
            # 1. 火电机组结果
            thermal_results = []
            for unit in self.model.ThermalUnits:
                for t in self.model.T:
                    thermal_results.append({
                        '机组': unit,
                        '时间': t,
                        '状态': pyo.value(self.model.thermal_status[unit, t]),
                        '出力': pyo.value(self.model.thermal_power[unit, t])
                    })
            thermal_df = pd.DataFrame(thermal_results)
            
            # 2. 可再生能源结果
            wind_results = []
            for unit in self.model.WindUnits:
                for t in self.model.T:
                    wind_results.append({
                        '机组': unit,
                        '时间': t,
                        '出力': pyo.value(self.model.wind_power[unit, t])
                    })
            wind_df = pd.DataFrame(wind_results)
            
            # 3. 光伏结果
            solar_results = []
            for unit in self.model.SolarUnits:
                for t in self.model.T:
                    solar_results.append({
                        '机组': unit,
                        '时间': t,
                        '出力': pyo.value(self.model.solar_power[unit, t])
                    })
            solar_df = pd.DataFrame(solar_results)
            
            # 4. 系统总结
            total_cost = pyo.value(self.model.total_cost)
            thermal_power_total = thermal_df['出力'].sum()
            wind_power_total = wind_df['出力'].sum()
            solar_power_total = solar_df['出力'].sum()
            
            summary_data = {
                '总成本': [total_cost],
                '总火电发电量': [thermal_power_total],
                '总风电发电量': [wind_power_total],
                '总光伏发电量': [solar_power_total]
            }
            summary_df = pd.DataFrame(summary_data)
            
            # 写入Excel
            with pd.ExcelWriter(output_path) as writer:
                thermal_df.to_excel(writer, sheet_name='火电机组结果', index=False)
                wind_df.to_excel(writer, sheet_name='风电结果', index=False)
                solar_df.to_excel(writer, sheet_name='光伏结果', index=False)
                summary_df.to_excel(writer, sheet_name='系统总结', index=False)
            
            logging.info(f"结果已保存到 {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"保存结果失败: {str(e)}")
            return None

# 主程序
if __name__ == "__main__":
    # 输入输出路径
    input_file = r"C:\Users\admin\Desktop\Rots.xlsx"
    output_file = "UC_Results.xlsx"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到输入文件 {input_file}")
        exit(1)

    try:
        # 创建模型
        uc_model = UC_Model(input_file)
        
        # 求解模型
        if uc_model.solve():
            # 保存结果
            result_path = uc_model.save_results(output_file)
            if result_path:
                print(f"优化完成，结果已保存到: {output_file}")
            else:
                print("结果保存失败")
        else:
            print("优化求解失败")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")