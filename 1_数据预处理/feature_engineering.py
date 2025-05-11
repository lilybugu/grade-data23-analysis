"""
格瑞德地源热泵系统特征工程
- 计算COP（性能系数）
- 计算供回水温差
- 计算热交换效率
- 添加时间特征
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 定义输入和输出目录
INPUT_DIR = './output'
OUTPUT_DIR = './output'

def load_data():
    """
    加载清洗后的数据
    """
    try:
        # 尝试加载pickle格式的数据（更快）
        data_path = os.path.join(INPUT_DIR, '清洗后数据.pkl')
        if os.path.exists(data_path):
            print(f"从 {data_path} 加载数据...")
            df = pd.read_pickle(data_path)
        else:
            # 如果没有pickle文件，加载CSV
            data_path = os.path.join(INPUT_DIR, '清洗后数据.csv')
            print(f"从 {data_path} 加载数据...")
            df = pd.read_csv(data_path)
            df['时间'] = pd.to_datetime(df['时间'])
        
        print(f"成功加载数据，形状: {df.shape}")
        return df
    
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def add_time_features(df):
    """
    添加时间相关特征
    """
    print("添加时间特征...")
    
    # 确保时间列是datetime类型
    if not pd.api.types.is_datetime64_dtype(df['时间']):
        df['时间'] = pd.to_datetime(df['时间'])
    
    # 提取各种时间特征
    df['月份'] = df['时间'].dt.month
    df['日期'] = df['时间'].dt.day
    df['小时'] = df['时间'].dt.hour
    df['工作日'] = df['时间'].dt.dayofweek < 5  # 0-4是工作日，5-6是周末
    
    # 创建季节特征
    df['季节'] = df['月份'].apply(lambda x: 
                                1 if x in [12, 1, 2] else   # 冬季
                                2 if x in [3, 4, 5] else   # 春季
                                3 if x in [6, 7, 8] else   # 夏季
                                4)                          # 秋季
    
    # 创建时段特征
    df['时段'] = df['小时'].apply(lambda x: 
                                1 if 0 <= x < 6 else    # 凌晨
                                2 if 6 <= x < 12 else   # 上午
                                3 if 12 <= x < 18 else  # 下午
                                4)                       # 晚上
    
    print("时间特征添加完成")
    return df

def calculate_cop_features(df):
    """
    计算COP相关特征
    """
    print("计算COP相关特征...")
    
    # 检查是否存在必要的列
    required_cols = []
    
    # 检查功率列
    power_cols = [col for col in df.columns if '功率' in col]
    if not power_cols:
        print("警告：未找到功率相关列，无法计算COP")
        return df
    else:
        required_cols.extend(power_cols)
    
    # 检查流量和温度列
    flow_cols = [col for col in df.columns if '流量' in col]
    temp_cols = [col for col in df.columns if '温度' in col]
    
    if not flow_cols or not temp_cols:
        print("警告：未找到流量或温度相关列，无法计算某些COP特征")
    else:
        required_cols.extend(flow_cols)
        required_cols.extend(temp_cols)
    
    # 确保所有必要的列都存在
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"警告：以下列不存在，部分COP计算可能受影响: {missing_cols}")
    
    # 计算各种能效指标
    try:
        # 1. 找到总功率列
        total_power_col = next((col for col in df.columns if '总有功功率' in col), None)
        
        # 2. 找到热量列
        heat_col = next((col for col in df.columns if '瞬时热流量' in col), None)
        
        # 3. 找到供回水温度列
        supply_temp_cols = [col for col in df.columns if '供水温度' in col]
        return_temp_cols = [col for col in df.columns if '回水温度' in col or '进水温度' in col]
        
        # 计算供回水温差
        if supply_temp_cols and return_temp_cols:
            for supply_col in supply_temp_cols:
                system_name = supply_col.split('温度')[0]
                matching_return_cols = [col for col in return_temp_cols if system_name in col]
                
                if matching_return_cols:
                    return_col = matching_return_cols[0]
                    temp_diff_col = f"{system_name}供回水温差"
                    df[temp_diff_col] = df[supply_col] - df[return_col]
                    print(f"已计算: {temp_diff_col}")
        
        # 计算COP（性能系数）
        if total_power_col and heat_col:
            df['COP'] = df[heat_col] / df[total_power_col]
            # 处理无穷大和NaN值
            df['COP'] = df['COP'].replace([np.inf, -np.inf], np.nan)
            df['COP'] = df['COP'].fillna(df['COP'].median())
            # 限制COP在合理范围内
            df.loc[df['COP'] < 0, 'COP'] = 0
            df.loc[df['COP'] > 10, 'COP'] = 10  # 假设COP最高不超过10
            
            print("已计算: COP (性能系数)")
        else:
            print("警告：未找到计算COP所需的列")
        
        # 计算能效比（如果有热量和电量数据）
        heat_energy_col = next((col for col in df.columns if '热量' in col and '累积' in col), None)
        electric_energy_col = next((col for col in df.columns if '电度' in col), None)
        
        if heat_energy_col and electric_energy_col:
            df['能效比'] = df[heat_energy_col].diff() / df[electric_energy_col].diff()
            # 处理无穷大和NaN值
            df['能效比'] = df['能效比'].replace([np.inf, -np.inf], np.nan)
            df['能效比'] = df['能效比'].fillna(df['能效比'].median())
            # 限制能效比在合理范围内
            df.loc[df['能效比'] < 0, '能效比'] = 0
            df.loc[df['能效比'] > 10, '能效比'] = 10
            
            print("已计算: 能效比")
    
    except Exception as e:
        print(f"计算COP相关特征时出错: {e}")
    
    print("COP相关特征计算完成")
    return df

def calculate_operation_features(df):
    """
    计算运行状态相关特征
    """
    print("计算运行状态特征...")
    
    # 运行模式
    mode_col = next((col for col in df.columns if '运行模式' in col), None)
    if mode_col:
        df['制热模式'] = df[mode_col] == 1
        df['制冷模式'] = df[mode_col] == 0
        print("已添加: 制热模式/制冷模式 标记")
    
    # 主机运行时长
    runtime_cols = [col for col in df.columns if '运行时长' in col]
    if runtime_cols:
        df['总运行时长'] = df[runtime_cols].sum(axis=1)
        print("已计算: 总运行时长")
    
    # 计算设备启停次数
    power_cols = [col for col in df.columns if '功率' in col and '总' not in col]
    for col in power_cols:
        if col in df.columns:
            # 设备名称
            device_name = col.split('功率')[0]
            # 状态变化列
            state_change_col = f"{device_name}启停状态变化"
            # 根据功率判断设备是否运行（功率>5认为在运行）
            df[f"{device_name}运行状态"] = df[col] > 5
            # 计算状态变化
            df[state_change_col] = df[f"{device_name}运行状态"].diff().abs()
            # 累计启停次数
            df[f"{device_name}累计启停次数"] = df[state_change_col].cumsum()
            print(f"已计算: {device_name}累计启停次数")
    
    # 计算平均运行周期（如果有足够数据）
    try:
        for col in power_cols:
            device_name = col.split('功率')[0]
            state_col = f"{device_name}运行状态"
            
            if state_col in df.columns:
                # 计算运行周期
                run_periods = []
                current_state = None
                current_start = None
                
                for idx, row in df.iterrows():
                    state = row[state_col]
                    
                    # 状态变化或首次设置状态
                    if state != current_state or current_state is None:
                        # 如果之前状态是"运行"，记录这个运行周期
                        if current_state == True and current_start is not None:
                            run_periods.append(idx - current_start)
                        
                        current_state = state
                        if state == True:  # 开始新的运行周期
                            current_start = idx
                
                # 计算平均运行周期长度
                if run_periods:
                    avg_run_period = sum(run_periods) / len(run_periods)
                    df[f"{device_name}平均运行周期"] = avg_run_period
                    print(f"已计算: {device_name}平均运行周期")
    except Exception as e:
        print(f"计算运行周期时出错: {e}")
    
    print("运行状态特征计算完成")
    return df

def calculate_environmental_impact(df):
    """
    计算环境影响和节能指标
    """
    print("计算环境影响和节能指标...")
    
    try:
        # 计算能耗强度（单位时间内的能耗）
        power_col = next((col for col in df.columns if '总有功功率' in col), None)
        if power_col:
            # 计算每小时平均能耗
            df['小时平均能耗'] = df.groupby(['月份', '日期', '小时'])[power_col].transform('mean')
            
            # 计算不同时段的能耗模式
            df['工作日能耗'] = df.groupby(['工作日', '小时'])[power_col].transform('mean')
            print("已计算: 小时平均能耗, 工作日能耗")
        
        # 能源利用效率（如果有热量数据）
        heat_col = next((col for col in df.columns if '热流量' in col), None)
        if heat_col and power_col:
            df['能源利用效率'] = df[heat_col] / df[power_col]
            df['能源利用效率'] = df['能源利用效率'].replace([np.inf, -np.inf], np.nan).fillna(0)
            print("已计算: 能源利用效率")
        
        # 计算碳排放（假设电力排放因子为0.5 kgCO2/kWh）
        if power_col:
            emission_factor = 0.5  # kgCO2/kWh
            df['碳排放量'] = df[power_col] * emission_factor
            print("已计算: 碳排放量")
        
        # 计算节能率（相对于固定基准）
        if 'COP' in df.columns:
            baseline_cop = 2.5  # 假设行业基准COP为2.5
            df['节能率'] = (df['COP'] - baseline_cop) / baseline_cop * 100
            print("已计算: 节能率 (相对于COP基准2.5)")
    
    except Exception as e:
        print(f"计算环境影响指标时出错: {e}")
    
    print("环境影响和节能指标计算完成")
    return df

def visualize_features(df):
    """
    可视化关键特征
    """
    print("生成特征可视化...")
    
    # 创建可视化输出目录
    viz_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    # 设置绘图风格
    plt.style.use('ggplot')
    
    # 1. COP随时间变化
    if 'COP' in df.columns:
        plt.figure(figsize=(12, 6))
        df.set_index('时间')['COP'].plot()
        plt.title('COP随时间变化')
        plt.ylabel('COP值')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'COP随时间变化.png'), dpi=300)
        print("已生成: COP随时间变化.png")
    
    # 2. 各月份平均COP
    if 'COP' in df.columns and '月份' in df.columns:
        plt.figure(figsize=(10, 6))
        monthly_cop = df.groupby('月份')['COP'].mean()
        monthly_cop.plot(kind='bar')
        plt.title('各月份平均COP')
        plt.ylabel('COP值')
        plt.xlabel('月份')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, '各月份平均COP.png'), dpi=300)
        print("已生成: 各月份平均COP.png")
    
    # 3. 功率与温度的关系
    power_col = next((col for col in df.columns if '总有功功率' in col), None)
    temp_col = next((col for col in df.columns if '环境温度' in col), None)
    
    if power_col and temp_col:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[temp_col], df[power_col], alpha=0.5)
        plt.title('环境温度与功率关系')
        plt.xlabel('环境温度')
        plt.ylabel('功率 (kW)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, '环境温度与功率关系.png'), dpi=300)
        print("已生成: 环境温度与功率关系.png")
    
    # 4. 供回水温差分布
    temp_diff_cols = [col for col in df.columns if '温差' in col]
    if temp_diff_cols:
        plt.figure(figsize=(12, 6))
        for col in temp_diff_cols[:3]:  # 最多绘制前三个温差列
            sns.kdeplot(df[col].dropna(), label=col)
        plt.title('供回水温差分布')
        plt.xlabel('温差 (°C)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, '供回水温差分布.png'), dpi=300)
        print("已生成: 供回水温差分布.png")
    
    # 5. 特征相关性热图
    if 'COP' in df.columns:
        # 选择数值型列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 从中选择最重要的特征（包括COP和相关列）
        important_cols = [col for col in numeric_cols if any(keyword in col for keyword in 
                                                       ['COP', '功率', '温度', '流量', '温差'])]
        # 限制特征数量，避免热图过大
        if len(important_cols) > 15:
            important_cols = important_cols[:15]
        
        if important_cols:
            plt.figure(figsize=(14, 12))
            corr_matrix = df[important_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('特征相关性热图')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, '特征相关性热图.png'), dpi=300)
            print("已生成: 特征相关性热图.png")
    
    print("特征可视化完成")

def main():
    """
    主函数
    """
    print("开始特征工程...")
    
    # 1. 加载清洗后的数据
    df = load_data()
    if df is None:
        print("数据加载失败，特征工程终止")
        return
    
    # 2. 添加时间特征
    df = add_time_features(df)
    
    # 3. 计算COP相关特征
    df = calculate_cop_features(df)
    
    # 4. 计算运行状态特征
    df = calculate_operation_features(df)
    
    # 5. 计算环境影响和节能指标
    df = calculate_environmental_impact(df)
    
    # 6. 可视化关键特征
    visualize_features(df)
    
    # 7. 保存处理后的特征工程数据
    df.to_csv(os.path.join(OUTPUT_DIR, '特征工程后数据.csv'), index=False, encoding='utf-8-sig')
    df.to_pickle(os.path.join(OUTPUT_DIR, '特征工程后数据.pkl'))
    print(f"特征工程后的数据已保存至 {os.path.join(OUTPUT_DIR, '特征工程后数据.csv')}")
    print(f"特征工程后的数据(pickle格式)已保存至 {os.path.join(OUTPUT_DIR, '特征工程后数据.pkl')}")
    
    # 8. 打印特征总结
    print("\n特征工程完成！")
    print(f"原始特征数: {len(df.columns) - len(df.select_dtypes(include=['datetime64']).columns)}")
    print(f"工程后特征总数: {len(df.columns) - len(df.select_dtypes(include=['datetime64']).columns)}")
    print(f"新增特征: {len(df.columns) - len(df.select_dtypes(include=['datetime64']).columns) - (len(df.columns) - len(df.select_dtypes(include=['datetime64']).columns))}")

if __name__ == "__main__":
    main() 