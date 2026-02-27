import pandas as pd

def check_hydro_data(file):
    try:
        df = pd.read_excel(file, sheet_name='UnitRunoffHydroGenerators')
        errors = []
        
        # 显示实际的列名，便于调试
        print(f"实际的列名: {list(df.columns)}")
        
        # 基础检查
        if df.isnull().values.any(): 
            errors.append("存在空值")
            
        # 检查是否有 Capacity 列（考虑空格）
        if 'Capacity ' not in df.columns:
            errors.append("缺少 'Capacity' 列")
        else:
            if not (df['Capacity '] > 0).all(): 
                errors.append("容量值无效")
                
        # 检查是否有 AreaName 列
        if 'AreaName' not in df.columns:
            errors.append("缺少 'AreaName' 列")
        else:
            if len(set(df['AreaName'])) != 3: 
                errors.append("区域数量错误")
            
        return errors if errors else "✅ 所有检查通过"
    except Exception as e:
        return f"❌ 文件读取失败: {str(e)}"

print(check_hydro_data('Final_Cleaned_Rots_v2.xlsx'))