import gurobipy as gp
from gurobipy import GRB

try:
    # 创建空模型
    model = gp.Model("Installation_Test")

    # 添加变量
    x = model.addVar(vtype=GRB.CONTINUOUS, name="x")
    y = model.addVar(vtype=GRB.CONTINUOUS, name="y")

    # 设置目标函数
    model.setObjective(x + y, GRB.MAXIMIZE)

    # 添加约束
    model.addConstr(x + 2 * y <= 4, "c0")
    model.addConstr(2 * x + y <= 4, "c1")

    # 优化求解
    model.optimize()

    # 检查求解状态
    if model.status == GRB.OPTIMAL:
        print("\nGurobi安装验证成功！")
        print(f"最优解: x={x.X:.2f}, y={y.X:.2f}")
        print(f"目标值 = {model.ObjVal:.2f}")
    else:
        print("求解未达到最优状态")

except gp.GurobiError as e:
    print(f"Gurobi错误: {e.message}")
except Exception as e:
    print(f"验证失败: {str(e)}")
    print("请检查: 1. Gurobi许可证是否激活 2. 是否安装gurobipy包")