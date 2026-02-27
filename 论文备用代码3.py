import pyomo.environ as pyo
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# 原始数据（不改网络/负荷/风电）
# --------------------------
GEN_BASE = {
    'G1': {'Pmax': 90, 'Pmin': 0, 'bus': 1, 'b': 40.0, 'c': 0.045, 'SU': 100.0},
    'G2': {'Pmax': 90, 'Pmin': 0, 'bus': 2, 'b': 40.0, 'c': 0.036, 'SU': 100.0},
    'G3': {'Pmax': 50, 'Pmin': 0, 'bus': 3, 'b': 30.0, 'c': 0.025, 'SU': 100.0},
}

DEMAND = [30, 80, 110, 50]
WIND_TOTAL = [6.02, 18.52, 11.82, 50.0]
WIND_SPLIT = {1: 0.5, 2: 0.5, 3: 0.0}

LINES = {
    'L12': {'i': 1, 'j': 2, 'x': 0.1, 'Fmax': 55.0},
    'L13': {'i': 1, 'j': 3, 'x': 0.1, 'Fmax': 55.0},
    'L23': {'i': 2, 'j': 3, 'x': 0.1, 'Fmax': 29.0},
}

T_PEAK = 2
LINE_KEY = 'L23'
THETA_GRID = [0, 50, 100, 200, 400]

# 目标对齐（论文 Fig.7 大致）
TARGET_G2_TH0   = 73.0
TARGET_G2_TH400 = 66.0

# reserve（保留；但它不是主矛盾）
C_UP, C_DN = 5.0, 4.0
ALPHA_UP, ALPHA_DN = 0.20, 0.10

# overflow penalty：先用线性（更接近论文“$/MW”直觉）
USE_LINEAR_OVERFLOW = True


def build_model(theta_over, mult_g2=1.0, mult_g3=1.0, mult_g1=1.0):
    # 只通过倍率调整成本方向（不改网络结构）
    GEN = {
        'G1': {**GEN_BASE['G1']},
        'G2': {**GEN_BASE['G2']},
        'G3': {**GEN_BASE['G3']},
    }
    GEN['G1']['b'] *= mult_g1
    GEN['G1']['c'] *= mult_g1
    GEN['G2']['b'] *= mult_g2
    GEN['G2']['c'] *= mult_g2
    GEN['G3']['b'] *= mult_g3
    GEN['G3']['c'] *= mult_g3

    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=range(4))
    m.G = pyo.Set(initialize=list(GEN.keys()))
    m.B = pyo.Set(initialize=[1, 2, 3])
    m.L = pyo.Set(initialize=list(LINES.keys()))

    m.Pmax = pyo.Param(m.G, initialize={g: GEN[g]['Pmax'] for g in m.G})
    m.Pmin = pyo.Param(m.G, initialize={g: GEN[g]['Pmin'] for g in m.G})
    m.bus  = pyo.Param(m.G, initialize={g: GEN[g]['bus']  for g in m.G})
    m.b    = pyo.Param(m.G, initialize={g: GEN[g]['b']    for g in m.G})
    m.c    = pyo.Param(m.G, initialize={g: GEN[g]['c']    for g in m.G})
    m.SU   = pyo.Param(m.G, initialize={g: GEN[g]['SU']   for g in m.G})

    m.demand = pyo.Param(m.T, initialize={t: DEMAND[t] for t in m.T})
    m.wind_total = pyo.Param(m.T, initialize={t: WIND_TOTAL[t] for t in m.T})
    m.wind_share = pyo.Param(m.B, initialize={b: float(WIND_SPLIT.get(b, 0.0)) for b in m.B})

    m.i = pyo.Param(m.L, initialize={l: LINES[l]['i'] for l in m.L})
    m.j = pyo.Param(m.L, initialize={l: LINES[l]['j'] for l in m.L})
    m.x = pyo.Param(m.L, initialize={l: LINES[l]['x'] for l in m.L})
    m.Fmax = pyo.Param(m.L, initialize={l: LINES[l]['Fmax'] for l in m.L})

    m.theta_over = pyo.Param(initialize=float(theta_over))

    # vars
    m.p = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)
    m.u = pyo.Var(m.G, m.T, within=pyo.Binary)
    m.y = pyo.Var(m.G, m.T, within=pyo.Binary)

    m.theta = pyo.Var(m.B, m.T, within=pyo.Reals)
    m.f = pyo.Var(m.L, m.T, within=pyo.Reals)
    m.of = pyo.Var(m.L, m.T, within=pyo.NonNegativeReals)

    m.ru = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)
    m.rd = pyo.Var(m.G, m.T, within=pyo.NonNegativeReals)

    # p-u linking
    m.c_pmax = pyo.Constraint(m.G, m.T, rule=lambda m,g,t: m.p[g,t] <= m.Pmax[g]*m.u[g,t])
    m.c_pmin = pyo.Constraint(m.G, m.T, rule=lambda m,g,t: m.p[g,t] >= m.Pmin[g]*m.u[g,t])

    # startup (assume on at t=0)
    u0 = {g: 1 for g in GEN}
    m.c_start = pyo.Constraint(
        m.G, m.T,
        rule=lambda m,g,t: m.y[g,t] >= (m.u[g,t] - (u0[g] if t==0 else m.u[g,t-1]))
    )

    # DC flow
    m.c_slack = pyo.Constraint(m.T, rule=lambda m,t: m.theta[3,t] == 0.0)
    m.c_flow  = pyo.Constraint(m.L, m.T, rule=lambda m,l,t: m.f[l,t] == (m.theta[m.i[l],t]-m.theta[m.j[l],t]) / m.x[l])

    # system balance
    m.c_sys = pyo.Constraint(m.T, rule=lambda m,t: sum(m.p[g,t] for g in m.G) + m.wind_total[t] == m.demand[t])

    # nodal balance
    def kcl(m, b, t):
        gen_at_b = sum(m.p[g,t] for g in m.G if m.bus[g]==b)
        wind_at_b = m.wind_share[b] * m.wind_total[t]
        demand_at_b = m.demand[t] if b==3 else 0.0
        inj = gen_at_b + wind_at_b - demand_at_b

        net = 0.0
        for l in m.L:
            if m.i[l]==b: net += m.f[l,t]
            elif m.j[l]==b: net -= m.f[l,t]
        return inj == net
    m.c_kcl = pyo.Constraint(m.B, m.T, rule=kcl)

    # overflow definition
    m.c_of1 = pyo.Constraint(m.L, m.T, rule=lambda m,l,t: m.of[l,t] >=  m.f[l,t] - m.Fmax[l])
    m.c_of2 = pyo.Constraint(m.L, m.T, rule=lambda m,l,t: m.of[l,t] >= -m.f[l,t] - m.Fmax[l])

    # reserve coupling
    m.c_ru_cap = pyo.Constraint(m.G, m.T, rule=lambda m,g,t: m.p[g,t] + m.ru[g,t] <= m.Pmax[g]*m.u[g,t])
    m.c_rd_cap = pyo.Constraint(m.G, m.T, rule=lambda m,g,t: m.p[g,t] - m.rd[g,t] >= m.Pmin[g]*m.u[g,t])

    # reserve requirement (system-wide)
    def up_req(m, t):
        net_load = m.demand[t] - m.wind_total[t]
        return sum(m.ru[g,t] for g in m.G) >= ALPHA_UP * net_load
    def dn_req(m, t):
        net_load = m.demand[t] - m.wind_total[t]
        return sum(m.rd[g,t] for g in m.G) >= ALPHA_DN * net_load
    m.c_up = pyo.Constraint(m.T, rule=up_req)
    m.c_dn = pyo.Constraint(m.T, rule=dn_req)

    # objective
    def obj(m):
        gen_cost = sum(m.c[g]*m.p[g,t]**2 + m.b[g]*m.p[g,t] for g in m.G for t in m.T)
        startup  = sum(m.SU[g]*m.y[g,t] for g in m.G for t in m.T)
        reserve_cost = sum(C_UP*m.ru[g,t] + C_DN*m.rd[g,t] for g in m.G for t in m.T)

        of_peak = m.of[LINE_KEY, T_PEAK]
        overflow_cost = m.theta_over * (of_peak if USE_LINEAR_OVERFLOW else of_peak**2)
        return gen_cost + startup + reserve_cost + overflow_cost

    m.obj = pyo.Objective(rule=obj, sense=pyo.minimize)
    return m, GEN


def solve_once(theta, mult_g2, mult_g3, mult_g1):
    m, GEN = build_model(theta, mult_g2=mult_g2, mult_g3=mult_g3, mult_g1=mult_g1)
    solver = pyo.SolverFactory('gurobi')
    solver.options['OutputFlag'] = 0
    solver.solve(m, tee=False)

    g2 = pyo.value(m.p['G2', T_PEAK])
    overflow = pyo.value(m.of[LINE_KEY, T_PEAK])

    # paper-style generating cost: exclude overflow penalty
    gen_cost_only = sum(pyo.value(m.c[g]*m.p[g,t]**2 + m.b[g]*m.p[g,t]) for g in GEN for t in range(4))
    startup_only  = sum(pyo.value(m.SU[g]*m.y[g,t]) for g in GEN for t in range(4))
    reserve_only  = sum(pyo.value(C_UP*m.ru[g,t] + C_DN*m.rd[g,t]) for g in GEN for t in range(4))
    total_gen_cost = gen_cost_only + startup_only + reserve_only

    return g2, overflow, total_gen_cost


def autofit():
    # 搜索范围：让 G2 变“更便宜”(mult_g2<1)，让 G3 变“更贵”(mult_g3>1)
    g2_grid = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    g3_grid = [1.00, 1.20, 1.40, 1.60, 1.80, 2.00, 2.30]
    g1_grid = [1.00]  # 先不动G1，避免自由度太多；需要再放开

    best = None
    for mg2 in g2_grid:
        for mg3 in g3_grid:
            for mg1 in g1_grid:
                g2_0, of_0, cost_0 = solve_once(0,   mg2, mg3, mg1)
                g2_4, of_4, cost_4 = solve_once(400, mg2, mg3, mg1)

                # 约束趋势方向：希望 cost 上升、G2 下降但不塌
                if not (cost_4 > cost_0):
                    continue
                if not (g2_0 > g2_4):
                    continue

                err = (g2_0 - TARGET_G2_TH0)**2 + (g2_4 - TARGET_G2_TH400)**2
                # 轻微约束：theta大时 overflow 不应太大（否则不像论文）
                err += 0.2 * (of_4**2)

                if best is None or err < best['err']:
                    best = {
                        'err': err,
                        'mg2': mg2, 'mg3': mg3, 'mg1': mg1,
                        'g2_0': g2_0, 'of_0': of_0, 'cost_0': cost_0,
                        'g2_4': g2_4, 'of_4': of_4, 'cost_4': cost_4
                    }

    if best is None:
        raise RuntimeError("没有找到同时满足 cost上升 且 G2下降 的倍率组合。请扩大搜索范围。")
    return best


if __name__ == "__main__":
    best = autofit()
    print("\n" + "="*110)
    print("AUTO-FIT 最佳倍率（让趋势方向与论文一致：θ↑ => overflow↓, G2↓, gen_cost↑）")
    print("="*110)
    print(best)

    mg2, mg3, mg1 = best['mg2'], best['mg3'], best['mg1']

    print("\n" + "="*96)
    print("扫 theta_over：输出 G2(t=2), overflow(L23,t2), gen_cost(不含惩罚项)")
    print("="*96)
    print(f"{'theta':>8} {'G2(t=2)':>10} {'overflow(L23,t2)':>18} {'gen_cost($)':>12}")
    print("-"*96)

    rows = []
    for th in THETA_GRID:
        g2, of, cost = solve_once(th, mg2, mg3, mg1)
        rows.append({'theta': th, 'G2_t2': g2, 'overflow_L23_t2': of, 'gen_cost': cost})
        print(f"{th:>8.0f} {g2:>10.2f} {of:>18.2f} {cost:>12.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("fig7_autofit_best.csv", index=False)
    print("\n✅ 保存结果：fig7_autofit_best.csv")

    # plot
    fig, ax1 = plt.subplots(figsize=(9.0, 4.6))
    ax1.plot(df['theta'], df['G2_t2'], marker='o', linewidth=2, label='G2(t=2)')
    ax1.plot(df['theta'], df['overflow_L23_t2'], marker='^', linestyle='--', linewidth=2, label='Overflow(L23,t2)')
    ax1.set_xlabel("Cost coefficient ($/MW)")
    ax1.set_ylabel("MW")
    ax1.grid(True, alpha=0.3, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(df['theta'], df['gen_cost'], marker='s', linewidth=2, label='Gen Cost')
    ax2.set_ylabel("System generating cost ($)")

    plt.title("Fig.7 replication (auto-fit costs)")
    plt.tight_layout()
    plt.savefig("fig7_autofit_best.png", dpi=250, bbox_inches="tight")
    plt.show()
