"""
IEEE.py 基础版本 - 仅包含最基本的约束，确保可行性
"""

# 创建基础版本的约束系统，逐步添加复杂性

def create_basic_model_constraints(self):
    """创建基础版本的模型约束"""
    
    # 1. 基本火电机组约束
    def thermal_logic_rule(model, unit, t):
        min_power = self.thermal_dict[unit].get('MinPower', 0)
        capacity = self.thermal_dict[unit].get('Capacity', 0)
        return model.thermal_power[unit, t] <= model.thermal_status[unit, t] * capacity
    
    def thermal_power_lower_rule(model, unit, t):
        min_power = self.thermal_dict[unit].get('MinPower', 0)
        return model.thermal_power[unit, t] >= model.thermal_status[unit, t] * min_power
    
    # 2. 基本可再生能源约束
    def wind_power_rule(model, unit, t):
        capacity = self.wind_dict[unit].get('Capacity', 0)
        return model.wind_power[unit, t] <= capacity
    
    def solar_power_rule(model, unit, t):
        capacity = self.solar_dict[unit].get('Capacity', 0)
        return model.solar_power[unit, t] <= capacity
    
    def hydro_power_rule(model, unit, t):
        capacity = self.hydro_dict[unit].get('Capacity', 0)
        return model.hydro_power[unit, t] <= capacity
    
    # 3. 基本储能约束
    def ess_charge_power_rule(model, unit, t):
        max_charge = self.ess_dict[unit].get('MaxChargePower', 0)
        return model.ess_charge[unit, t] <= max_charge
    
    def ess_discharge_power_rule(model, unit, t):
        max_discharge = self.ess_dict[unit].get('MaxDisPower', 0)
        return model.ess_discharge[unit, t] <= max_discharge
    
    # 4. 全局功率平衡约束
    def global_power_balance_rule(model, t):
        thermal_gen = sum(model.thermal_power[unit, t] for unit in model.ThermalUnits)
        wind_gen = sum(model.wind_power[unit, t] for unit in model.WindUnits)
        solar_gen = sum(model.solar_power[unit, t] for unit in model.SolarUnits)
        hydro_gen = sum(model.hydro_power[unit, t] for unit in model.HydroUnits)
        ess_discharge = sum(model.ess_discharge[unit, t] for unit in model.ESSUntis)
        ess_charge = sum(model.ess_charge[unit, t] for unit in model.ESSUntis)
        
        if t < len(self.load_curve):
            total_load = self.load_curve[t]
        else:
            total_load = 0
        
        total_shed = sum(model.load_shed[load, t] for load in model.Loads)
        
        return (thermal_gen + wind_gen + solar_gen + hydro_gen + ess_discharge ==
                total_load - total_shed + ess_charge)
    
    return {
        'thermal_logic': thermal_logic_rule,
        'thermal_power_lower': thermal_power_lower_rule,
        'wind_power': wind_power_rule,
        'solar_power': solar_power_rule,
        'hydro_power': hydro_power_rule,
        'ess_charge_power': ess_charge_power_rule,
        'ess_discharge_power': ess_discharge_power_rule,
        'global_power_balance': global_power_balance_rule
    }
