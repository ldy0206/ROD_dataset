import pandas as pd
import itertools

print("1. 正在读取切分好的数据...")
p_train = pd.read_csv('P_train.tsv', sep='\t')
p_test = pd.read_csv('P_test.tsv', sep='\t')
n_train = pd.read_csv('N_train.tsv', sep='\t')
n_test = pd.read_csv('N_test.tsv', sep='\t')

# 把训练集的正负样本合并，测试集的正负样本合并
train_df = pd.concat([p_train, n_train], ignore_index=True)
test_df = pd.concat([p_test, n_test], ignore_index=True)

# 为了保证独热编码(One-hot)时，训练集和测试集的疾病种类完全对齐，我们先把它俩拼在一起处理
all_df = pd.concat([train_df, test_df], ignore_index=True)

print("2. 正在提取疾病特征 (One-hot 独热编码)...")
# 根据 MeSH_ID 生成独热编码，前缀设置为 Disease
disease_features = pd.get_dummies(all_df['MeSH_ID'], prefix='Disease')
# 将生成的疾病特征重新拼接到总表里
all_df = pd.concat([all_df, disease_features], axis=1)

print("3. 正在提取 miRNA 序列特征 (请耐心等待，这可能需要十几秒)...")
# 准备 3-mer 的 64 种组合 (AAA, AAC ... UUU)
bases = ['A', 'C', 'G', 'U']
kmers = [''.join(p) for p in itertools.product(bases, repeat=3)]

# 定义一个处理单条序列的函数
def extract_seq_features(seq):
    seq = str(seq).upper().replace('T', 'U') # 统一转大写，并把可能存在的T转为U
    length = len(seq)
    
    # 基础特征
    if length == 0:
        return [0] * (6 + 64) # 防止空序列报错
        
    a_prop = seq.count('A') / length
    c_prop = seq.count('C') / length
    g_prop = seq.count('G') / length
    u_prop = seq.count('U') / length
    gc_content = (seq.count('G') + seq.count('C')) / length
    
    # k-mer 特征 (k=3)
    kmer_freqs = []
    k = 3
    total_kmers = length - k + 1
    if total_kmers > 0:
        for kmer in kmers:
            # 计算每个 3-mer 出现的次数并除以总可能数
            count = sum(1 for i in range(total_kmers) if seq[i:i+k] == kmer)
            kmer_freqs.append(count / total_kmers)
    else:
        kmer_freqs = [0] * 64

    # 返回所有特征的列表：长度, A, C, G, U, GC含量, 64种k-mer频率
    return [length, a_prop, c_prop, g_prop, u_prop, gc_content] + kmer_freqs

# 定义序列特征的列名
seq_col_names = ['Seq_Length', 'Prop_A', 'Prop_C', 'Prop_G', 'Prop_U', 'GC_Content'] + [f'3mer_{k}' for k in kmers]

# 将函数应用到每一行数据上，生成包含所有序列特征的表格
seq_features_df = pd.DataFrame(all_df['Sequence'].apply(extract_seq_features).tolist(), columns=seq_col_names)

# 把序列特征拼接到总表里
all_df = pd.concat([all_df, seq_features_df], axis=1)

print("4. 正在拆分并保存最终数据集...")
# 把那些对机器训练没用的文字列删掉（留着 label）
columns_to_drop = ['RNA_ID', 'Sequence', 'MeSH_ID', 'MeSH_Name', 'PMID', 'Year']
# 仅保留存在于表中的列去删除（因为阴性数据没有 PMID 和 Year）
columns_to_drop = [c for c in columns_to_drop if c in all_df.columns]
final_df = all_df.drop(columns=columns_to_drop)

# 把总表重新拆分回训练集和测试集
final_train = final_df.iloc[:len(train_df)]
final_test = final_df.iloc[len(train_df):]

# 导出最终的数据集
final_train.to_csv('train_dataset.tsv', sep='\t', index=False)
final_test.to_csv('test_dataset.tsv', sep='\t', index=False)

print("✅ 特征提取完成！")
print(f"最终训练集大小: {final_train.shape[0]} 行, {final_train.shape[1]} 列 (特征维度)")
print(f"最终测试集大小: {final_test.shape[0]} 行, {final_test.shape[1]} 列 (特征维度)")