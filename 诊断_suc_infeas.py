#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断SUC不可行性的原因"""

from 论文第二部分测试 import IEEERTS79System
import numpy as np

system = IEEERTS79System()
print(f'N_GEN: {system.N_GEN}')
print(f'N_BUS: {system.N_BUS}')
print(f'N_BRANCH: {system.N_BRANCH}')
print(f'load_profile shape: {np.array(system.load_profile).shape}')
print(f'\nload_profile[0] (first hour): {system.load_profile[0]}')
print(f'load_profile[0] type: {type(system.load_profile[0])}')
print(f'load_profile[0] is list: {isinstance(system.load_profile[0], list)}')
if isinstance(system.load_profile[0], list):
    print(f'load_profile[0] length: {len(system.load_profile[0])}')

print(f'\nwind_farms columns: {system.wind_farms.columns.tolist()}')
print(f'generators shape: {system.generators.shape}')

Pmin_sum = system.generators['Pmin'].sum()
Pmax_sum = system.generators['Pmax'].sum()
print(f'Pmin sum: {Pmin_sum}')
print(f'Pmax sum: {Pmax_sum}')

wind_caps = system.wind_farms['capacity'].sum()
print(f'Wind capacity sum: {wind_caps}')

if isinstance(system.load_profile[0], list):
    load_hour_0 = sum(system.load_profile[0])
else:
    load_hour_0 = system.load_profile[0].sum() if hasattr(system.load_profile[0], 'sum') else sum(system.load_profile[0])
    
print(f'Total load hour 0: {load_hour_0}')

# 检查风电场景
scenarios = system.get_wind_scenarios(3)
print(f'\nWind scenarios count: {len(scenarios)}')
print(f'Scenario 0 shape: {np.array(scenarios[0]).shape}')
print(f'Scenario 0[0]: {scenarios[0][0]}')
print(f'Scenario 0[0] type: {type(scenarios[0][0])}')
if isinstance(scenarios[0][0], list) or isinstance(scenarios[0][0], np.ndarray):
    wind_hour_0 = sum(scenarios[0][0])
else:
    wind_hour_0 = scenarios[0][0]
print(f'Total wind hour 0 scenario 0: {wind_hour_0}')

print(f'\n可行性检查:')
print(f'Pmin + Wind >= Load? {Pmin_sum} + {wind_hour_0} >= {load_hour_0}? {Pmin_sum + wind_hour_0 >= load_hour_0}')
print(f'Pmax >= Load? {Pmax_sum} >= {load_hour_0}? {Pmax_sum >= load_hour_0}')
