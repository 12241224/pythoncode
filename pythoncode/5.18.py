import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference

# 读取Excel文件（如果不存在会新建）
excel_file = "data.xlsx"
sheet_name = "Sheet1"

# 示例数据（如果文件不存在）
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
    "Sales": [200, 300, 250, 400, 350],
    "Cost": [150, 200, 180, 220, 210]
}

# 创建/读取DataFrame并写入Excel
df = pd.DataFrame(data)
df.to_excel(excel_file, index=False, sheet_name=sheet_name)

# 使用openpyxl加载工作簿
wb = load_workbook(excel_file)
ws = wb[sheet_name]

# 创建折线图
chart = LineChart()
chart.title = "Sales & Cost Trend"
chart.style = 13  # 预定义样式
chart.y_axis.title = "Amount"
chart.x_axis.title = "Month"

# 设置数据范围（从A2到C6）
data = Reference(ws, min_col=1, min_row=2, max_col=3, max_row=6)
categories = Reference(ws, min_col=1, min_row=2, max_row=6)  # 月份作为分类轴

# 添加数据到图表
chart.add_data(data, titles_from_data=True)
chart.set_categories(categories)

# 将图表添加到工作表（位置从E2开始）
ws.add_chart(chart, "E2")

# 保存修改后的Excel文件
wb.save(excel_file)
print(f"图表已生成并保存到 {excel_file}")