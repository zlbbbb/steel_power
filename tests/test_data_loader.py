# 查看数据文件中的实际列名
import pandas as pd
data = pd.read_csv('data/processed/final/final_processed.csv')
print("可用列:", data.columns.tolist())