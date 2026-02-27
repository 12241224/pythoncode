#!/usr/bin/env python
# -*- coding: utf-8 -*-

from 论文第二部分测试 import IEEERTS79System
import numpy as np

system = IEEERTS79System()
print(f'Branch capacities (first 10): {system.branches["capacity"].values[:10]}')
avg_capacity = np.mean(system.branches['capacity'])
print(f'Average capacity: {avg_capacity:.1f}')
print(f'Limit (3.5x avg): {avg_capacity * 3.5:.1f}')
print(f'Total Pmax: {system.generators["Pmax"].sum():.1f}')

# 检查每小时的负荷
for t in [0, 6, 12, 18, 23]:
    load_t = sum(system.load_profile[t])
    print(f'Hour {t} load: {load_t:.1f}')
