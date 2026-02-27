# 调试RUC参数计算
import numpy as np
from scipy import stats

class CCVLinearizer:
    """CCV（Constrained Cost Variable）分段线性化工具类"""
    
    def __init__(self, n_segments=10):
        self.n_segments = n_segments
    
    def linearize_eens(self, D_t, sigma_t, R_up_range=None):
        """线性化EENS（期望失负荷）"""
        # 如果没有指定范围，使用3倍标准差
        if R_up_range is None:
            R_up_range = [0, 3 * sigma_t]
        
        # 生成分段点
        segments = np.linspace(R_up_range[0], R_up_range[1], self.n_segments + 1)
        
        print(f"D_t={D_t}, sigma_t={sigma_t}")
        print(f"R_up_range={R_up_range}")
        print(f"segments={segments}")
        
        a_params = []
        b_params = []
        
        # 计算每个分段点的EENS值
        R_up_values = []
        eens_values = []
        
        for R_up in segments:
            # 计算z值
            z = (R_up - D_t) / sigma_t if sigma_t > 0 else 0
            
            # 计算EENS：EENS = σ[φ(z) - z(1-Φ(z))]
            if sigma_t > 0:
                phi = stats.norm.pdf(z)
                Phi = stats.norm.cdf(z)
                eens = sigma_t * (phi - z * (1 - Phi))
            else:
                eens = 0
            
            print(f"  R_up={R_up:.0f}, z={z:.4f}, phi={phi:.6f}, Phi={Phi:.6f}, eens={eens:.4f}")
            
            R_up_values.append(R_up)
            eens_values.append(eens)
        
        # 计算分段线性参数
        for k in range(self.n_segments):
            R_up_k = R_up_values[k]
            R_up_k1 = R_up_values[k + 1]
            eens_k = eens_values[k]
            eens_k1 = eens_values[k + 1]
            
            # 计算斜率和截距
            if R_up_k1 != R_up_k:
                a_k = (eens_k1 - eens_k) / (R_up_k1 - R_up_k)
            else:
                a_k = 0
            
            b_k = eens_k - a_k * R_up_k
            
            a_params.append(a_k)
            b_params.append(b_k)
        
        print(f"a_params={a_params}")
        print(f"b_params={b_params}")
        
        return a_params, b_params

# 测试
ccv = CCVLinearizer(n_segments=10)

# 测试第一个小时的参数
total_load = 2850 * 0.68  # 第一小时的负荷
total_wind = 180 * 0.5 * 2  # 两个风电场各50%
D_t = total_load - total_wind
wind_std = total_wind * 0.15
sigma_t = wind_std

print("="*50)
print("第一个小时的EENS线性化参数")
print("="*50)
a, b = ccv.linearize_eens(D_t, sigma_t)

print("\n" + "="*50)
print("计算约束中的EENS值")
print("="*50)
# 测试一些上备用值
for R_up in [0, 100, 200, 300]:
    # 从约束 R_ens >= a_k * R_up + b_k（对所有k）
    max_eens = max([a_k * R_up + b_k for a_k, b_k in zip(a, b)])
    print(f"R_up={R_up}: R_ens >= {max_eens:.4f}")
