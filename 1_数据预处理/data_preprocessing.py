"""
格瑞德地源热泵系统数据预处理
第一步：数据预处理阶段
- 将所有CSV文件合并成一个完整的宽表
- 检查数据质量，处理缺失值和异常值
- 统一时间戳格式，确保数据时间对齐
"""

import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import logging

# 确保输出目录存在
OUTPUT_DIR = './output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "数据预处理日志.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 定义数据目录 - 修改为当前目录的父目录，即数据所在位置
DATA_DIR = '../..'  # 修改为实际数据目录

def load_all_csv_files():
    """
    加载所有CSV文件并合并成一个宽表
    """
    # 获取所有CSV文件路径
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    logging.info(f"找到 {len(csv_files)} 个CSV文件")
    
    if len(csv_files) == 0:
        logging.error(f"在目录 {os.path.abspath(DATA_DIR)} 下未找到任何CSV文件，请检查路径设置！")
        return pd.DataFrame()  # 返回空DataFrame
    
    # 创建一个DataFrame用于存储时间列
    all_data = pd.DataFrame()
    
    # 首先处理时间列
    first_file = True
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        logging.info(f"正在处理文件: {filename}")
        
        # 读取CSV文件
        try:
            # 尝试不同的编码方式
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
            
            # 处理列名为'time'和'last'的情况
            if 'time' in df.columns and 'last' in df.columns:
                # 重命名列
                # 用文件名作为数据列名（去掉.csv扩展名）
                sensor_name = os.path.splitext(filename)[0]
                df.rename(columns={'time': '时间', 'last': sensor_name}, inplace=True)
                
                # 将时间列转换为日期时间格式
                df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
                
                # 合并数据
                if all_data.empty:
                    all_data = df.copy()
                else:
                    # 合并数据，根据时间列进行合并
                    all_data = pd.merge(all_data, df, on='时间', how='outer')
            # 检查是否有时间列（保留原有处理逻辑作为备选）
            elif '时间' in df.columns:
                # 确保时间格式一致
                df['时间'] = pd.to_datetime(df['时间'], errors='coerce')
                
                # 重命名列名，添加文件名作为前缀，避免重复列名
                # 先提取出传感器名称作为列名前缀
                sensor_name = filename.replace('.csv', '')
                
                # 对数据列重命名
                value_cols = [col for col in df.columns if col != '时间']
                if len(value_cols) == 1:
                    # 如果只有一个数据列，直接用传感器名称
                    df.rename(columns={value_cols[0]: sensor_name}, inplace=True)
                else:
                    # 如果有多个数据列，则加上列名
                    for col in value_cols:
                        df.rename(columns={col: f"{sensor_name}_{col}"}, inplace=True)
                
                # 合并数据
                if all_data.empty:
                    all_data = df
                else:
                    all_data = pd.merge(all_data, df, on='时间', how='outer')
            else:
                logging.warning(f"警告：文件 {filename} 的列名异常，跳过该文件")
        except Exception as e:
            logging.error(f"处理文件 {filename} 时出错: {str(e)}")
    
    return all_data

def clean_data(df):
    """
    清洗数据，处理缺失值和异常值
    """
    logging.info("开始数据清洗...")
    logging.info(f"原始数据维度: {df.shape}")
    
    # 如果DataFrame为空，直接返回
    if df.empty:
        logging.error("数据为空，无法进行清洗。请检查数据加载步骤。")
        return df
    
    # 计算每列的缺失值比例
    missing_percentage = df.isnull().sum() * 100 / len(df)
    logging.info("\n各列缺失值百分比:")
    for col, percent in zip(missing_percentage.index, missing_percentage.values):
        logging.info(f"{col}: {percent:.2f}%")
    
    # 对于缺失值过多的列（例如超过80%），可以选择删除
    columns_to_drop = missing_percentage[missing_percentage > 80].index.tolist()
    if columns_to_drop:
        logging.info(f"\n删除缺失值过多的列: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    
    # 处理缺失值
    missing_cols = df.columns[df.isnull().any()]
    for col in missing_cols:
        if df[col].dtype in [np.float64, np.int64]:
            # 对于数值型数据，根据缺失比例决定填充方法
            if df[col].isnull().sum() / len(df) < 0.3:  # 如果缺失值少于30%
                logging.info(f"对列 {col} 使用插值填充")
                df[col] = df[col].interpolate(method='linear').ffill().bfill()
            else:
                # 对于大量缺失的数据，可以考虑使用每小时的中位数或平均值填充
                logging.info(f"对列 {col} 使用时间模式填充")
                # 提取小时
                df['hour'] = df['时间'].dt.hour
                # 计算每小时的中位数
                hourly_median = df.groupby('hour')[col].transform('median')
                # 使用小时中位数填充
                df[col] = df[col].fillna(hourly_median)
                # 如果仍有缺失，使用列中位数填充
                df[col] = df[col].fillna(df[col].median())
                # 删除临时小时列
                df.drop('hour', axis=1, inplace=True)
        else:
            # 对于非数值型数据，使用前向和后向填充
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # 检测并处理异常值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == '时间':
            continue
        
        # 计算IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 检测异常值
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            logging.info(f"列 {col} 中发现 {len(outliers)} 个异常值")
            
            # 将异常值替换为边界值
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound
    
    logging.info(f"清洗后数据维度: {df.shape}")
    return df

def analyze_data_quality(df):
    """
    分析数据质量，生成数据质量报告
    """
    logging.info("\n生成数据质量报告...")
    
    # 如果DataFrame为空，直接返回
    if df.empty:
        logging.error("数据为空，无法生成质量报告。")
        return pd.DataFrame()
    
    # 计算基本统计量
    stats = df.describe().T
    stats['missing_rate'] = df.isnull().mean() * 100
    
    # 使用时间戳作为文件名后缀，避免文件被锁定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f'数据质量报告_{timestamp}.csv'
    
    # 保存统计报告
    stats.to_csv(os.path.join(OUTPUT_DIR, report_filename), encoding='utf-8-sig')
    logging.info(f"数据质量报告已保存至 {os.path.join(OUTPUT_DIR, report_filename)}")
    
    # 绘制时间序列数据质量热图
    vis_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    # 可视化缺失值
    plt.figure(figsize=(12, 8))
    plt.title('数据缺失情况热图')
    heatmap_data = df.isnull()
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='缺失')
    plt.xlabel('列索引')
    plt.ylabel('行索引')
    
    # 使用时间戳命名图片文件
    viz_filename = f'缺失值热图_{timestamp}.png'
    plt.savefig(os.path.join(vis_dir, viz_filename), dpi=300, bbox_inches='tight')
    
    return stats

def main():
    """
    主函数
    """
    logging.info("开始数据预处理...")
    
    # 1. 加载并合并所有CSV文件
    merged_data = load_all_csv_files()
    
    # 如果数据为空，提前退出
    if merged_data.empty:
        logging.error("合并后的数据为空，程序终止。请检查数据目录和文件格式。")
        return
    
    # 2. 数据清洗，处理缺失值和异常值
    cleaned_data = clean_data(merged_data)
    
    # 3. 分析数据质量
    data_quality_stats = analyze_data_quality(cleaned_data)
    
    # 使用时间戳作为文件名后缀，避免文件被锁定
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 4. 保存处理后的数据
    csv_filename = f'清洗后数据_{timestamp}.csv'
    cleaned_data.to_csv(os.path.join(OUTPUT_DIR, csv_filename), index=False, encoding='utf-8-sig')
    logging.info(f"清洗后的数据已保存至 {os.path.join(OUTPUT_DIR, csv_filename)}")
    
    # 5. 保存一份pickle格式的数据，便于后续分析
    pkl_filename = f'清洗后数据_{timestamp}.pkl'
    cleaned_data.to_pickle(os.path.join(OUTPUT_DIR, pkl_filename))
    logging.info(f"清洗后的数据(pickle格式)已保存至 {os.path.join(OUTPUT_DIR, pkl_filename)}")
    
    # 同时保存一个固定名称的最新版本，方便后续程序引用
    cleaned_data.to_csv(os.path.join(OUTPUT_DIR, '清洗后数据_最新.csv'), index=False, encoding='utf-8-sig')
    cleaned_data.to_pickle(os.path.join(OUTPUT_DIR, '清洗后数据_最新.pkl'))
    
    logging.info("数据预处理完成！")

if __name__ == "__main__":
    main() 