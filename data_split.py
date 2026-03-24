import pandas as pd

# 1. 读取 Excel 文件中的两个工作表
print("正在读取 Excel 数据，请稍候...")
# 注意：确保这里的文件名和你的 Excel 文件名一字不差
file_name = 'dataset.xlsx'
df_pos = pd.read_excel(file_name, sheet_name='positive')
df_neg = pd.read_excel(file_name, sheet_name='negative')

# 2. 挑选出需要的列（保留了拼写的 'label'）
pos_cols = ['RNA_ID', 'Sequence', 'MeSH_ID', 'MeSH_Name', 'PMID', 'Year', 'label']
neg_cols = ['RNA_ID', 'Sequence', 'MeSH_ID', 'MeSH_Name', 'label']

# 过滤掉这些列中如果有空白的数据行，保证数据干净
df_pos = df_pos[pos_cols].dropna(subset=['Sequence', 'MeSH_ID'])
df_neg = df_neg[neg_cols].dropna(subset=['Sequence', 'MeSH_ID'])

# 3. 阳性数据：按年份切分
print("正在按年份切分阳性数据...")
# 训练集：年份小于等于 2024
p_train = df_pos[df_pos['Year'] <= 2024]
# 测试集：年份等于 2025 （包含2025及以后的新数据）
p_test = df_pos[df_pos['Year'] >= 2025] 

# 4. 阴性数据：按 1:1 比例随机抽样
print("正在为阴性数据按 1:1 比例抽样...")
n_train_count = len(p_train) # 训练集正样本数量
n_test_count = len(p_test)   # 测试集正样本数量

# 设定 random_state=42 是为了保证每次抽样结果一样，方便复现
# 给训练集抽样
n_train = df_neg.sample(n=n_train_count, random_state=42)
# 把抽走的数据从总池子里剔除，保证训练集和测试集的负样本绝对不重合
df_neg_remaining = df_neg.drop(n_train.index)
# 从剩下的池子里给测试集抽样
n_test = df_neg_remaining.sample(n=n_test_count, random_state=42)

# 5. 保存成 4 个方案要求的 TSV 格式文件
print("正在导出文件...")
# 分别导出训练集和测试集文件
p_train.to_csv('P_train.tsv', sep='\t', index=False)
p_test.to_csv('P_test.tsv', sep='\t', index=False)
n_train.to_csv('N_train.tsv', sep='\t', index=False)
n_test.to_csv('N_test.tsv', sep='\t', index=False)

print("✅ 处理完成！")
print(f"训练集: 包含 {len(p_train)} 个阳性，{len(n_train)} 个阴性。")
print(f"测试集: 包含 {len(p_test)} 个阳性，{len(n_test)} 个阴性。")