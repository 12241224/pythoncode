#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断PTDF矩阵中的NaN问题"""

from 论文第二部分测试 import IEEERTS79System
import numpy as np

system = IEEERTS79System()

print('支路reactance统计:')
print(f'  最小: {system.branches["reactance"].min()}')
print(f'  最大: {system.branches["reactance"].max()}')
print(f'  平均: {system.branches["reactance"].mean()}')
print(f'  零值数: {(system.branches["reactance"] == 0).sum()}')

# 重新计算Bbus
print('\n重新计算Bbus...')
n_bus = system.N_BUS
Bbus = np.zeros((n_bus, n_bus))
for idx, branch in system.branches.iterrows():
    from_bus = int(branch['from_bus']) - 1
    to_bus = int(branch['to_bus']) - 1
    reactance = float(branch['reactance'])
    
    if reactance <= 0:
        print(f'Warning: 支路 {idx} 的reactance={reactance}')
        continue
        
    susceptance = 1.0 / reactance
    Bbus[from_bus, from_bus] += susceptance
    Bbus[to_bus, to_bus] += susceptance
    Bbus[from_bus, to_bus] -= susceptance
    Bbus[to_bus, from_bus] -= susceptance

print('Bbus矩阵对角线前5个:', np.diag(Bbus)[:5])
print(f'Bbus中的NaN: {np.isnan(Bbus).sum()}')
print(f'Bbus中的inf: {np.isinf(Bbus).sum()}')
print(f'Bbus矩阵秩: {np.linalg.matrix_rank(Bbus)}')

# 尝试inversion
print('\n尝试矩阵求逆...')
slack_node = n_bus - 1
non_slack_idx = list(range(slack_node)) + list(range(slack_node + 1, n_bus))
Bbus_dc = Bbus[np.ix_(non_slack_idx, non_slack_idx)]

print(f'Bbus_dc形状: {Bbus_dc.shape}')
print(f'Bbus_dc中的NaN: {np.isnan(Bbus_dc).sum()}')
print(f'Bbus_dc秩: {np.linalg.matrix_rank(Bbus_dc)}')

try:
    Bbus_dc_inv = np.linalg.inv(Bbus_dc)
    print('✓ 矩阵求逆成功')
    print(f'求逆矩阵中的NaN: {np.isnan(Bbus_dc_inv).sum()}')
except Exception as e:
    print(f'✗ 矩阵求逆失败: {e}')
    Bbus_dc_inv = np.linalg.pinv(Bbus_dc)
    print(f'使用pinv，NaN: {np.isnan(Bbus_dc_inv).sum()}')
