"""
RUC 数据输入整合模块（Excel 版）

用途：
1) 将原始硬编码输入数据集中管理
2) 尽可能将输入参数存入 Excel 多工作表
3) 从 Excel 读取并还原为原模型可直接使用的数据字典结构
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


DEFAULT_EXCEL_NAME = "RUC输入数据.xlsx"


def get_convex_segments(points: List[Tuple[float, float]], num_segments: int = 10) -> Dict[str, List[float]]:
    """根据曲线控制点生成分段线性参数 y = a*x + b。"""
    sorted_points = sorted((float(x), float(y)) for x, y in points)
    x_vals = [p[0] for p in sorted_points]
    y_vals = [p[1] for p in sorted_points]

    x_samples = np.linspace(min(x_vals), max(x_vals), num_segments + 1)
    y_samples = np.interp(x_samples, x_vals, y_vals)

    segments = {"a": [], "b": []}
    for idx in range(num_segments):
        x1, y1 = x_samples[idx], y_samples[idx]
        x2, y2 = x_samples[idx + 1], y_samples[idx + 1]
        a_val = 0.0 if abs(x2 - x1) < 1e-9 else (y2 - y1) / (x2 - x1)
        b_val = y1 - a_val * x1
        segments["a"].append(float(a_val))
        segments["b"].append(float(b_val))

    return segments


def build_default_input_data() -> Dict[str, Any]:
    """构建与原代码兼容的默认输入数据。"""

    generators = {
        "G1": {"Pmax": 90, "Pmin": 10, "a": 40, "b": 450, "c_up": 5, "c_down": 5, "startup": 100, "ramp": 90},
        "G2": {"Pmax": 90, "Pmin": 10, "a": 30, "b": 250, "c_up": 4, "c_down": 4, "startup": 100, "ramp": 90},
        "G3": {"Pmax": 50, "Pmin": 10, "a": 40, "b": 360, "c_up": 4, "c_down": 4, "startup": 100, "ramp": 50},
    }

    buses = ["Bus1", "Bus2", "Bus3"]

    branches = {
        "Line1-2": {"from": "Bus1", "to": "Bus2", "reactance": 0.1, "capacity": 55, "G": [0.0000, -0.6667, -0.3333]},
        "Line1-3": {"from": "Bus1", "to": "Bus3", "reactance": 0.1, "capacity": 55, "G": [0.0000, -0.3333, -0.6667]},
        "Line2-3": {"from": "Bus2", "to": "Bus3", "reactance": 0.1, "capacity": 55, "G": [0.0000, 0.3333, -0.3333]},
    }

    gen_bus = {"G1": "Bus1", "G2": "Bus2", "G3": "Bus3"}

    wind_farms = {
        "WF1": {"bus": "Bus1", "capacity": 50},
        "WF2": {"bus": "Bus2", "capacity": 50},
    }

    time_periods = [1, 2, 3, 4]

    load_data = {
        "Bus1": [0, 0, 0, 0],
        "Bus2": [0, 0, 0, 0],
        "Bus3": [30, 80, 110, 50],
    }

    wind_forecast = {
        "WF1": [3.01, 4.26, 5.91, 8.39],
        "WF2": [3.01, 4.26, 5.91, 8.39],
    }

    wind_std = {
        "WF1": [0.6, 0.8, 1.2, 1.6],
        "WF2": [0.6, 0.8, 1.2, 1.6],
    }

    risk_penalties = {
        "load_shedding": 50,
        "wind_curtailment": 50,
        "branch_overflow": 50,
    }

    eens_points = {
        1: [(0, 30), (2, 8), (4, 1), (20, 0.1)],
        2: [(0, 25), (3, 10), (6, 2), (20, 0.1)],
        3: [(0, 20), (5, 12), (10, 4), (20, 0.5)],
        4: [(0, 15), (6, 11), (12, 7), (20, 2.0)],
    }

    wc_points = {
        1: [(0, 5), (3, 1), (10, 0.1), (20, 0.0)],
        2: [(0, 15), (5, 8), (10, 2), (15, 0.5), (20, 0)],
        3: [(0, 25), (5, 15), (10, 6), (15, 1.5), (20, 0.2)],
        4: [(0, 35), (5, 20), (10, 8), (15, 2), (20, 0.5)],
    }

    piecewise_params = {"EENS": {}, "WindCurt": {}}
    for t in time_periods:
        piecewise_params["EENS"][t] = get_convex_segments(eens_points[t])
        piecewise_params["WindCurt"][t] = get_convex_segments(wc_points[t])

    return {
        "generators": generators,
        "buses": buses,
        "branches": branches,
        "gen_bus": gen_bus,
        "wind_farms": wind_farms,
        "time_periods": time_periods,
        "load_data": load_data,
        "wind_forecast": wind_forecast,
        "wind_std": wind_std,
        "risk_penalties": risk_penalties,
        "piecewise_params": piecewise_params,
        "curve_points": {"EENS": eens_points, "WindCurt": wc_points},
    }


def _dict_series_to_long(data_map: Dict[str, List[float]], periods: List[int], value_name: str) -> pd.DataFrame:
    rows = []
    for key, values in data_map.items():
        for t_idx, t_val in enumerate(periods):
            rows.append({"id": key, "t": int(t_val), value_name: float(values[t_idx])})
    return pd.DataFrame(rows)


def _curve_points_to_df(curve_map: Dict[int, List[Tuple[float, float]]]) -> pd.DataFrame:
    rows = []
    for t_val, pts in curve_map.items():
        for x_val, y_val in pts:
            rows.append({"t": int(t_val), "x": float(x_val), "y": float(y_val)})
    return pd.DataFrame(rows)


def export_data_to_excel(data: Dict[str, Any], excel_path: str) -> None:
    """将输入数据导出到 Excel 多工作表。"""

    buses = data["buses"]
    periods = data["time_periods"]

    generators_df = pd.DataFrame.from_dict(data["generators"], orient="index").reset_index().rename(columns={"index": "gen"})

    branch_rows = []
    for line, row in data["branches"].items():
        item = {
            "line": line,
            "from": row["from"],
            "to": row["to"],
            "reactance": float(row["reactance"]),
            "capacity": float(row["capacity"]),
        }
        for idx, bus in enumerate(buses):
            item[f"G_{bus}"] = float(row["G"][idx])
        branch_rows.append(item)
    branches_df = pd.DataFrame(branch_rows)

    gen_bus_df = pd.DataFrame(
        [{"gen": gen, "bus": bus} for gen, bus in data["gen_bus"].items()]
    )
    wind_farms_df = pd.DataFrame.from_dict(data["wind_farms"], orient="index").reset_index().rename(columns={"index": "wf"})

    time_df = pd.DataFrame({"t": periods})
    buses_df = pd.DataFrame({"bus": buses})

    load_df = _dict_series_to_long(data["load_data"], periods, "load")
    load_df = load_df.rename(columns={"id": "bus"})

    wind_forecast_df = _dict_series_to_long(data["wind_forecast"], periods, "forecast").rename(columns={"id": "wf"})
    wind_std_df = _dict_series_to_long(data["wind_std"], periods, "std").rename(columns={"id": "wf"})

    risk_df = pd.DataFrame(
        [{"item": key, "value": float(value)} for key, value in data["risk_penalties"].items()]
    )

    eens_points_df = _curve_points_to_df(data["curve_points"]["EENS"])
    windcurt_points_df = _curve_points_to_df(data["curve_points"]["WindCurt"])

    excel_file = Path(excel_path)
    excel_file.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        generators_df.to_excel(writer, sheet_name="generators", index=False)
        buses_df.to_excel(writer, sheet_name="buses", index=False)
        branches_df.to_excel(writer, sheet_name="branches", index=False)
        gen_bus_df.to_excel(writer, sheet_name="gen_bus", index=False)
        wind_farms_df.to_excel(writer, sheet_name="wind_farms", index=False)
        time_df.to_excel(writer, sheet_name="time_periods", index=False)
        load_df.to_excel(writer, sheet_name="load_data", index=False)
        wind_forecast_df.to_excel(writer, sheet_name="wind_forecast", index=False)
        wind_std_df.to_excel(writer, sheet_name="wind_std", index=False)
        risk_df.to_excel(writer, sheet_name="risk_penalties", index=False)
        eens_points_df.to_excel(writer, sheet_name="curve_EENS", index=False)
        windcurt_points_df.to_excel(writer, sheet_name="curve_WindCurt", index=False)


def _read_required_sheet(excel_data: Dict[str, pd.DataFrame], sheet_name: str) -> pd.DataFrame:
    if sheet_name not in excel_data:
        raise ValueError(f"Excel 缺少必需工作表: {sheet_name}")
    return excel_data[sheet_name].copy()


def load_data_from_excel(excel_path: str, num_segments: int = 10) -> Dict[str, Any]:
    """从 Excel 读取输入数据并还原成原模型兼容结构。"""

    excel_data = pd.read_excel(excel_path, sheet_name=None)

    generators_df = _read_required_sheet(excel_data, "generators")
    buses_df = _read_required_sheet(excel_data, "buses")
    branches_df = _read_required_sheet(excel_data, "branches")
    gen_bus_df = _read_required_sheet(excel_data, "gen_bus")
    wind_farms_df = _read_required_sheet(excel_data, "wind_farms")
    time_df = _read_required_sheet(excel_data, "time_periods")
    load_df = _read_required_sheet(excel_data, "load_data")
    wind_forecast_df = _read_required_sheet(excel_data, "wind_forecast")
    wind_std_df = _read_required_sheet(excel_data, "wind_std")
    risk_df = _read_required_sheet(excel_data, "risk_penalties")
    eens_points_df = _read_required_sheet(excel_data, "curve_EENS")
    windcurt_points_df = _read_required_sheet(excel_data, "curve_WindCurt")

    buses = [str(x) for x in buses_df["bus"].tolist()]
    time_periods = [int(x) for x in sorted(time_df["t"].tolist())]

    generators = {}
    for _, row in generators_df.iterrows():
        gen = str(row["gen"])
        generators[gen] = {
            "Pmax": float(row["Pmax"]),
            "Pmin": float(row["Pmin"]),
            "a": float(row["a"]),
            "b": float(row["b"]),
            "c_up": float(row["c_up"]),
            "c_down": float(row["c_down"]),
            "startup": float(row["startup"]),
            "ramp": float(row["ramp"]),
        }

    branches = {}
    ptdf_cols = [f"G_{bus}" for bus in buses]
    for _, row in branches_df.iterrows():
        line = str(row["line"])
        branches[line] = {
            "from": str(row["from"]),
            "to": str(row["to"]),
            "reactance": float(row["reactance"]),
            "capacity": float(row["capacity"]),
            "G": [float(row[col]) for col in ptdf_cols],
        }

    gen_bus = {str(row["gen"]): str(row["bus"]) for _, row in gen_bus_df.iterrows()}

    wind_farms = {}
    for _, row in wind_farms_df.iterrows():
        wf = str(row["wf"])
        wind_farms[wf] = {"bus": str(row["bus"]), "capacity": float(row["capacity"])}

    def _restore_series(df: pd.DataFrame, id_col: str, value_col: str) -> Dict[str, List[float]]:
        restored = {}
        for key, group_df in df.groupby(id_col):
            temp = group_df[["t", value_col]].sort_values("t")
            restored[str(key)] = [float(v) for v in temp[value_col].tolist()]
        return restored

    load_data = _restore_series(load_df, "bus", "load")
    wind_forecast = _restore_series(wind_forecast_df, "wf", "forecast")
    wind_std = _restore_series(wind_std_df, "wf", "std")

    risk_penalties = {str(row["item"]): float(row["value"]) for _, row in risk_df.iterrows()}

    def _restore_curve_points(df: pd.DataFrame) -> Dict[int, List[Tuple[float, float]]]:
        result = {}
        for t_val, group_df in df.groupby("t"):
            sorted_df = group_df.sort_values("x")
            result[int(t_val)] = [(float(xv), float(yv)) for xv, yv in zip(sorted_df["x"], sorted_df["y"])]
        return result

    eens_points = _restore_curve_points(eens_points_df)
    wc_points = _restore_curve_points(windcurt_points_df)

    piecewise_params = {"EENS": {}, "WindCurt": {}}
    for t in time_periods:
        piecewise_params["EENS"][t] = get_convex_segments(eens_points[t], num_segments=num_segments)
        piecewise_params["WindCurt"][t] = get_convex_segments(wc_points[t], num_segments=num_segments)

    return {
        "generators": generators,
        "buses": buses,
        "branches": branches,
        "gen_bus": gen_bus,
        "wind_farms": wind_farms,
        "time_periods": time_periods,
        "load_data": load_data,
        "wind_forecast": wind_forecast,
        "wind_std": wind_std,
        "risk_penalties": risk_penalties,
        "piecewise_params": piecewise_params,
        "curve_points": {
            "EENS": eens_points,
            "WindCurt": wc_points,
        },
    }


def extract_paper_data(excel_path: str | None = None, auto_create_template: bool = True) -> Dict[str, Any]:
    """
    对外兼容接口：优先从 Excel 读取；如果不存在则自动用默认数据生成模板后再读取。
    """

    if excel_path is None:
        excel_path = str(Path(__file__).with_name(DEFAULT_EXCEL_NAME))

    excel_file = Path(excel_path)

    if not excel_file.exists():
        if not auto_create_template:
            raise FileNotFoundError(f"未找到输入 Excel: {excel_path}")
        data = build_default_input_data()
        export_data_to_excel(data, str(excel_file))

    return load_data_from_excel(str(excel_file))


def create_excel_template(excel_path: str | None = None) -> str:
    """创建一份默认 Excel 输入模板。"""
    if excel_path is None:
        excel_path = str(Path(__file__).with_name(DEFAULT_EXCEL_NAME))

    data = build_default_input_data()
    export_data_to_excel(data, excel_path)
    return excel_path


if __name__ == "__main__":
    output_file = create_excel_template()
    print(f"已生成输入模板: {output_file}")
    loaded = extract_paper_data(output_file)
    print("读取验证通过:")
    print(f"  发电机数量: {len(loaded['generators'])}")
    print(f"  风电场数量: {len(loaded['wind_farms'])}")
    print(f"  时间段数: {len(loaded['time_periods'])}")
