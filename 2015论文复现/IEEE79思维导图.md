# IEEE RTS-79系统UC模型精确复现

## 1. 数据层 `IEEERTS79System`类
- **发电机参数**
  - 9种类型32台机组
  - `RTS79_GENERATORS`
- **网络拓扑**
  - 从PYPOWER获取
  - 24母线38支路
- **PTDF矩阵计算**
  - `_calculate_ptdf()`
- **负荷与风电数据**
  - `DAILY_LOAD_PATTERN`
  - `DAILY_WIND_PATTERN`

## 2. 工具层 `CCVLinearizer`类
- **分段线性化工具**
  - `linearize_eens` (EENS线性化)
  - `linearize_wind_curtailment` (弃风线性化)
  - `linearize_overflow` (支路越限线性化)
  - `linearize_quadratic_cost` (二次成本线性化)

## 3. 模型层 四种UC模型
- **DUC** (确定性UC)
  - 功率平衡约束
  - 机组出力/爬坡约束
  - 最小启停时间
  - 备用需求约束 (3%固定备用)
  - DC潮流约束 (PTDF)
- **RUC** (基于风险的UC)
  - 继承DUC全部约束
  - **预计算风险参数**
    - `_precompute_eens_params`
    - `_precompute_overflow_params`
  - **风险变量** (`R_ens`/`R_wind`/`R_flow`)
  - **CCV分段线性化约束** (EENS/弃风/越限)
  - **风险成本目标**
    - $\theta_{ens}=5000$
    - $\theta_{wind}=100$
    - $\theta_{flow}=500$
- **SUC1** (单场景UC)
  - 第一阶段 (U/SU变量)
  - 第二阶段 (P/R_up/R_down变量)
  - 松弛变量 (Load_Shed/Wind_Curt)
  - 场景概率 ($n=1$)
- **SUC2** (多场景UC)
  - 与SUC1结构相同
  - 场景数20个
  - **聚类生成场景** `get_wind_scenarios`
  - 全支路PTDF约束

## 4. 评估层 `evaluate_uc_performance`
- **统一的性能评估**
  - 3000次蒙特卡洛采样
- **矢量化计算**
  - 失负荷/弃风/越限
- **实际发电成本计算**
  - 燃料+启停
- **风险惩罚折算**
  - VOLL=5000$/MWh

## 5. 主程序层 `main`与可视化
- 运行四个模型
- 生成表IV对比数据
- 计算相对误差
- 可视化图表 (6个子图)
- 保存结果CSV生成思维导图
