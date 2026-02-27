"""
第三部分代码的数据输入整合（Excel版）

目标：
1. 将 RTS-79 第三部分脚本中的核心输入参数集中管理
2. 支持 Excel 多工作表维护输入数据
3. 提供“自动生成模板 + 读取配置”能力
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


DEFAULT_EXCEL_NAME = "第三部分_输入数据.xlsx"


def build_default_config() -> Dict[str, Any]:
    """构建与第三部分脚本兼容的默认输入配置。"""
    config = {
        "system_params": {
            "peak_load": 2850.0,
            "limited_line_capacity": 175.0,
            "normal_line_capacity": 500.0,
        },
        "RTS79_GENERATORS": [
            {
                "type": "nuclear_400", "count": 2, "Pmin": 100, "Pmax": 400,
                "cost_a": 0.0, "cost_b": 12.0, "cost_c": 800,
                "ramp_up": 100, "ramp_down": 100, "min_up": 8, "min_down": 8,
                "startup_cost": 3000, "buses": [22, 23],
            },
            {
                "type": "coal_350", "count": 1, "Pmin": 140, "Pmax": 350,
                "cost_a": 0.0, "cost_b": 18.0, "cost_c": 700,
                "ramp_up": 150, "ramp_down": 150, "min_up": 8, "min_down": 8,
                "startup_cost": 2500, "buses": [23],
            },
            {
                "type": "coal_197", "count": 3, "Pmin": 69, "Pmax": 197,
                "cost_a": 0.0, "cost_b": 22.0, "cost_c": 600,
                "ramp_up": 100, "ramp_down": 100, "min_up": 8, "min_down": 8,
                "startup_cost": 1800, "buses": [13, 15, 16],
            },
            {
                "type": "coal_155", "count": 4, "Pmin": 54, "Pmax": 155,
                "cost_a": 0.0, "cost_b": 24.0, "cost_c": 550,
                "ramp_up": 80, "ramp_down": 80, "min_up": 6, "min_down": 6,
                "startup_cost": 1500, "buses": [15, 16, 23, 1],
            },
            {
                "type": "oil_100", "count": 3, "Pmin": 25, "Pmax": 100,
                "cost_a": 0.0, "cost_b": 28.0, "cost_c": 500,
                "ramp_up": 50, "ramp_down": 50, "min_up": 4, "min_down": 4,
                "startup_cost": 1200, "buses": [7, 7, 13],
            },
            {
                "type": "coal_76", "count": 4, "Pmin": 15, "Pmax": 76,
                "cost_a": 0.0, "cost_b": 26.0, "cost_c": 450,
                "ramp_up": 40, "ramp_down": 40, "min_up": 4, "min_down": 4,
                "startup_cost": 1000, "buses": [1, 2, 2, 13],
            },
            {
                "type": "gas_50", "count": 6, "Pmin": 12, "Pmax": 50,
                "cost_a": 0.0, "cost_b": 32.0, "cost_c": 400,
                "ramp_up": 30, "ramp_down": 30, "min_up": 2, "min_down": 2,
                "startup_cost": 800, "buses": [3, 4, 5, 6, 9, 10],
            },
            {
                "type": "oil_20", "count": 4, "Pmin": 5, "Pmax": 20,
                "cost_a": 0.0, "cost_b": 38.0, "cost_c": 300,
                "ramp_up": 20, "ramp_down": 20, "min_up": 1, "min_down": 1,
                "startup_cost": 500, "buses": [11, 12, 14, 18],
            },
            {
                "type": "oil_12", "count": 5, "Pmin": 2, "Pmax": 12,
                "cost_a": 0.0, "cost_b": 42.0, "cost_c": 250,
                "ramp_up": 12, "ramp_down": 12, "min_up": 1, "min_down": 1,
                "startup_cost": 400, "buses": [19, 20, 21, 22, 24],
            },
        ],
        "RTS79_WINDMILLS": [
            {"bus": 14, "capacity": 340, "forecast_std": 0.15},
            {"bus": 19, "capacity": 340, "forecast_std": 0.15},
        ],
        "BUS_LOAD_DISTRIBUTION": {
            1: 0.038, 2: 0.034, 3: 0.063, 4: 0.026,
            5: 0.025, 6: 0.048, 7: 0.044, 8: 0.060,
            9: 0.061, 10: 0.068, 11: 0.030, 12: 0.030,
            13: 0.050, 14: 0.030, 15: 0.040, 16: 0.030,
            17: 0.030, 18: 0.025, 19: 0.050, 20: 0.050,
            21: 0.020, 22: 0.020, 23: 0.020, 24: 0.020,
        },
        "DAILY_LOAD_PATTERN": [
            0.68, 0.66, 0.64, 0.63, 0.62, 0.63, 0.65, 0.70,
            0.80, 0.90, 0.95, 0.98, 0.98, 0.97, 0.97, 0.95,
            0.93, 0.95, 0.98, 0.97, 0.92, 0.85, 0.78, 0.72,
        ],
        "DAILY_WIND_PATTERN": [
            0.85, 0.88, 0.90, 0.85, 0.75, 0.60, 0.45, 0.35,
            0.30, 0.25, 0.20, 0.25, 0.30, 0.35, 0.30, 0.25,
            0.30, 0.40, 0.55, 0.70, 0.80, 0.85, 0.90, 0.88,
        ],
        "LIMITED_LINES": [
            (11, 13), (12, 13), (14, 16), (15, 16),
            (15, 21), (15, 24), (16, 17), (18, 21),
        ],
    }
    return config


def _generators_to_df(generators: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in generators:
        row = item.copy()
        buses = row.pop("buses")
        row["buses"] = ",".join(str(int(b)) for b in buses)
        rows.append(row)
    return pd.DataFrame(rows)


def _generators_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    cols_int = ["count", "Pmin", "Pmax", "ramp_up", "ramp_down", "min_up", "min_down", "startup_cost"]
    cols_float = ["cost_a", "cost_b", "cost_c"]

    result: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        item = {
            "type": str(row["type"]),
            "count": int(row["count"]),
            "Pmin": float(row["Pmin"]),
            "Pmax": float(row["Pmax"]),
            "cost_a": float(row["cost_a"]),
            "cost_b": float(row["cost_b"]),
            "cost_c": float(row["cost_c"]),
            "ramp_up": float(row["ramp_up"]),
            "ramp_down": float(row["ramp_down"]),
            "min_up": int(row["min_up"]),
            "min_down": int(row["min_down"]),
            "startup_cost": float(row["startup_cost"]),
            "buses": [int(x.strip()) for x in str(row["buses"]).split(",") if x.strip()],
        }

        for key in cols_int:
            if key in item:
                item[key] = int(round(float(item[key])))
        for key in cols_float:
            if key in item:
                item[key] = float(item[key])

        result.append(item)
    return result


def export_config_to_excel(config: Dict[str, Any], excel_path: str) -> None:
    excel_file = Path(excel_path)
    excel_file.parent.mkdir(parents=True, exist_ok=True)

    generators_df = _generators_to_df(config["RTS79_GENERATORS"])
    wind_df = pd.DataFrame(config["RTS79_WINDMILLS"])

    bus_dist_df = pd.DataFrame(
        [{"bus": int(bus), "ratio": float(ratio)} for bus, ratio in config["BUS_LOAD_DISTRIBUTION"].items()]
    ).sort_values("bus")

    load_pattern_df = pd.DataFrame({"hour": list(range(1, 25)), "value": config["DAILY_LOAD_PATTERN"]})
    wind_pattern_df = pd.DataFrame({"hour": list(range(1, 25)), "value": config["DAILY_WIND_PATTERN"]})

    limited_lines_df = pd.DataFrame(
        [{"from_bus": int(a), "to_bus": int(b)} for a, b in config["LIMITED_LINES"]]
    )

    system_params_df = pd.DataFrame(
        [{"param": key, "value": float(value)} for key, value in config["system_params"].items()]
    )

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        generators_df.to_excel(writer, sheet_name="generators", index=False)
        wind_df.to_excel(writer, sheet_name="wind_farms", index=False)
        bus_dist_df.to_excel(writer, sheet_name="bus_load_distribution", index=False)
        load_pattern_df.to_excel(writer, sheet_name="daily_load_pattern", index=False)
        wind_pattern_df.to_excel(writer, sheet_name="daily_wind_pattern", index=False)
        limited_lines_df.to_excel(writer, sheet_name="limited_lines", index=False)
        system_params_df.to_excel(writer, sheet_name="system_params", index=False)


def _read_required_sheet(excel_data: Dict[str, pd.DataFrame], sheet_name: str) -> pd.DataFrame:
    if sheet_name not in excel_data:
        raise ValueError(f"Excel 缺少必需工作表: {sheet_name}")
    return excel_data[sheet_name].copy()


def load_config_from_excel(excel_path: str) -> Dict[str, Any]:
    excel_data = pd.read_excel(excel_path, sheet_name=None)

    generators_df = _read_required_sheet(excel_data, "generators")
    wind_df = _read_required_sheet(excel_data, "wind_farms")
    bus_dist_df = _read_required_sheet(excel_data, "bus_load_distribution")
    load_pattern_df = _read_required_sheet(excel_data, "daily_load_pattern")
    wind_pattern_df = _read_required_sheet(excel_data, "daily_wind_pattern")
    limited_lines_df = _read_required_sheet(excel_data, "limited_lines")
    system_params_df = _read_required_sheet(excel_data, "system_params")

    generators = _generators_from_df(generators_df)
    wind_farms = []
    for _, row in wind_df.iterrows():
        wind_farms.append(
            {
                "bus": int(row["bus"]),
                "capacity": float(row["capacity"]),
                "forecast_std": float(row["forecast_std"]),
            }
        )

    bus_dist = {int(row["bus"]): float(row["ratio"]) for _, row in bus_dist_df.iterrows()}
    total_dist = sum(bus_dist.values())
    if total_dist <= 0:
        raise ValueError("bus_load_distribution 的比例和必须大于0")
    bus_dist = {bus: ratio / total_dist for bus, ratio in bus_dist.items()}

    daily_load_pattern = [float(x) for x in load_pattern_df.sort_values("hour")["value"].tolist()]
    daily_wind_pattern = [float(x) for x in wind_pattern_df.sort_values("hour")["value"].tolist()]

    if len(daily_load_pattern) != 24 or len(daily_wind_pattern) != 24:
        raise ValueError("daily_load_pattern 和 daily_wind_pattern 必须各有24行")

    limited_lines: List[Tuple[int, int]] = []
    for _, row in limited_lines_df.iterrows():
        limited_lines.append((int(row["from_bus"]), int(row["to_bus"])))

    system_params = {str(row["param"]): float(row["value"]) for _, row in system_params_df.iterrows()}

    return {
        "system_params": system_params,
        "RTS79_GENERATORS": generators,
        "RTS79_WINDMILLS": wind_farms,
        "BUS_LOAD_DISTRIBUTION": bus_dist,
        "DAILY_LOAD_PATTERN": daily_load_pattern,
        "DAILY_WIND_PATTERN": daily_wind_pattern,
        "LIMITED_LINES": limited_lines,
    }


def load_or_create_config(excel_path: str | None = None, auto_create_template: bool = True) -> Dict[str, Any]:
    if excel_path is None:
        excel_path = str(Path(__file__).with_name(DEFAULT_EXCEL_NAME))

    excel_file = Path(excel_path)
    if not excel_file.exists():
        if not auto_create_template:
            raise FileNotFoundError(f"未找到输入 Excel: {excel_path}")
        default_config = build_default_config()
        export_config_to_excel(default_config, str(excel_file))

    return load_config_from_excel(str(excel_file))


def create_excel_template(excel_path: str | None = None) -> str:
    if excel_path is None:
        excel_path = str(Path(__file__).with_name(DEFAULT_EXCEL_NAME))

    config = build_default_config()
    export_config_to_excel(config, excel_path)
    return excel_path


if __name__ == "__main__":
    path = create_excel_template()
    print(f"已生成第三部分输入模板: {path}")
    loaded = load_or_create_config(path)
    print("读取校验通过:")
    print(f"  发电机类型数: {len(loaded['RTS79_GENERATORS'])}")
    print(f"  风电场数: {len(loaded['RTS79_WINDMILLS'])}")
    print(f"  负荷曲线点数: {len(loaded['DAILY_LOAD_PATTERN'])}")
