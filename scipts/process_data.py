import pandas as pd
import sys
import os
import time


# --- 设置 Python 路径 ---
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# --- 导入 DataLoader ---
from src.dataloader.dataloader import DataLoader

# --- 定义文件路径 ---
RAW_DATA_PATH = os.path.join(module_path, 'data', 'stocks.csv')
PROCESSED_DATA_PATH = os.path.join(module_path, 'data', 'stocks_processed.csv')

def main():
    print("开始加载原始数据...")
    start_time = time.time()
    
    try:
        # --- 加载原始数据 ---
        raw_data = pd.read_csv(RAW_DATA_PATH)
        print(f"原始数据加载完成，形状: {raw_data.shape}")
        print("\n原始数据预览:")
        print(raw_data.head())
        
        # --- 实例化 DataLoader ---
        print("\n开始数据预处理...")
        loader = DataLoader(raw_data)
        
        # --- 处理数据 ---
        processed_data = loader.preprocess_data()
        print(f"\n数据预处理完成，处理后形状: {processed_data.shape}")
        
        # --- 验证处理结果 ---
        print("\n处理结果验证:")
        print("1. 数据形状:", processed_data.shape)
        print("2. 索引名称:", processed_data.index.names)
        print("\n3. 缺失值统计:")
        print(processed_data.isnull().sum())
        
        # 检查几个关键因子的统计信息
        print("\n4. 关键因子统计信息:")
        factor_cols = ['ep_ratio', 'roe', 'pb_ratio']  # 根据实际列名调整
        for col in factor_cols:
            if col in processed_data.columns:
                print(f"\n{col} 统计:")
                print(processed_data[col].describe())
        
        # --- 保存处理后的数据 ---
        print(f"\n开始保存处理后的数据到 {PROCESSED_DATA_PATH}...")
        processed_data.to_csv(PROCESSED_DATA_PATH)  # 不保存自动生成的索引
        print("数据保存成功！")
        
        end_time = time.time()
        print(f"\n总耗时: {end_time - start_time:.2f} 秒")
        
    except FileNotFoundError:
        print(f"错误：找不到原始数据文件 {RAW_DATA_PATH}")
        sys.exit(1)
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()