#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查PYPOWER数据结构"""

from pypower.api import case24_ieee_rts
import numpy as np

ppc = case24_ieee_rts()

print("=== 发电机数据结构 ===")
print("Gen形状:", ppc['gen'].shape)
print("前2行发电机数据:")
print(ppc['gen'][:2])

print("\n=== 发电机表列定义（PYPOWER）===")
print("第0列（母线）:", ppc['gen'][:3, 0].astype(int))
print("第8列（Pmax）:", ppc['gen'][:3, 8])
print("第9列（Pmin）:", ppc['gen'][:3, 9])

# 支路表
print("\n=== 支路数据结构 ===")
print("Branch形状:", ppc['branch'].shape)
print("第5列（rateA - 容量）的样本:", ppc['branch'][:3, 5])

# 母线表
print("\n=== 母线数据结构 ===")
print("Bus形状:", ppc['bus'].shape)
print("前2行母线数据:")
print(ppc['bus'][:2])

# 成本表
print("\n=== Cost表 ===")
if 'gencost' in ppc:
    print("Gencost形状:", ppc['gencost'].shape)
    print("前2行成本数据:")
    print(ppc['gencost'][:2])
else:
    print("无gencost表")

# 检查负荷
print("\n=== 负荷数据 ===")
print("Pd列（有功负荷）在bus[:, 2]:")
print(ppc['bus'][:3, 2])

# 检查支路阻抗
print("\n=== 支路阻抗信息 ===")
print("第4列（impedance X）:" , ppc['branch'][:3, 4])
