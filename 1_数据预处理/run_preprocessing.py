"""
运行完整的数据预处理流程
"""

import os
import time
import subprocess
import sys
import glob
import logging
import pandas as pd

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def check_environment():
    """检查Python环境"""
    print_header("检查Python环境")
    
    # 检查是否在虚拟环境中
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("警告: 未检测到虚拟环境。建议在虚拟环境中运行此脚本。")
        print("可以通过运行 setup_environment.bat 来创建和激活虚拟环境。")
        print("继续运行可能会导致依赖包冲突。")
        
        # 询问是否继续
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            print("退出程序。")
            sys.exit(0)
    else:
        print("已检测到虚拟环境。")
        
    # 检查所需的包
    required_packages = ["pandas", "numpy", "matplotlib", "seaborn"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    if missing_packages:
        print("\n缺少必要的Python包。请安装：")
        print(f"pip install {' '.join(missing_packages)}")
        
        # 询问是否继续
        response = input("\n是否尝试自动安装这些包? (y/n): ")
        if response.lower() == 'y':
            for package in missing_packages:
                print(f"安装 {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} 安装完成。")
        else:
            print("请手动安装缺少的包后再运行此脚本。")
            sys.exit(1)
    
    print("\n环境检查完成。")

def create_output_directory():
    """创建输出目录"""
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")
    
    return output_dir

def run_data_preprocessing():
    """运行数据预处理脚本"""
    print_header("开始数据预处理")
    
    start_time = time.time()
    
    # 运行数据预处理脚本
    print("执行 data_preprocessing.py...")
    try:
        import data_preprocessing
        data_preprocessing.main()
    except Exception as e:
        print(f"执行数据预处理时出错: {str(e)}")
        return False
    
    end_time = time.time()
    print(f"\n数据预处理完成，耗时: {end_time - start_time:.2f} 秒")
    return True

def run_feature_engineering():
    """运行特征工程脚本"""
    print_header("开始特征工程")
    
    start_time = time.time()
    
    # 检查预处理文件是否存在
    if not os.path.exists("./output/清洗后数据.pkl") and not os.path.exists("./output/清洗后数据.csv"):
        print("错误: 未找到预处理数据文件。请先运行数据预处理步骤。")
        return False
    
    # 运行特征工程脚本
    print("执行 feature_engineering.py...")
    try:
        import feature_engineering
        feature_engineering.main()
    except Exception as e:
        print(f"执行特征工程时出错: {str(e)}")
        return False
    
    end_time = time.time()
    print(f"\n特征工程完成，耗时: {end_time - start_time:.2f} 秒")
    return True

def display_summary(output_dir):
    """显示处理结果摘要"""
    print_header("处理结果摘要")
    
    # 显示生成的文件
    print("生成的文件:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for file in files:
            size = os.path.getsize(os.path.join(root, file)) / 1024  # KB
            print(f"{subindent}{file} ({size:.1f} KB)")
    
    print("\n预处理和特征工程已完成。")
    print("您现在可以进行数据分析或构建机器学习模型。")

def main():
    """主函数"""
    print_header("地源热泵系统数据处理流程")
    
    # 检查环境
    check_environment()
    
    # 创建输出目录
    output_dir = create_output_directory()
    
    # 运行数据预处理
    preprocessing_success = run_data_preprocessing()
    
    # 如果预处理失败，不继续执行后续步骤
    if not preprocessing_success:
        print("数据预处理失败，程序终止。")
        return
    
    # 运行特征工程
    feature_engineering_success = run_feature_engineering()
    
    # 显示处理结果摘要
    if feature_engineering_success:
        display_summary(output_dir)
    
    print("\n所有处理步骤已完成。")
    print("下一步是进行数据分析，可以切换到'2_数据分析'目录。")

if __name__ == "__main__":
    main() 