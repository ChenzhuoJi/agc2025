import json
import pandas as pd
from scipy.sparse import coo_matrix
import datetime
from src.processor import GraphProcessingManager
import os
# ----------------edges------------------
edges = pd.read_csv(r"data\raw\lasftm_asia\lastfm_asia_edges.csv").to_numpy()
# ----------------features------------------
with open(r"data\raw\lasftm_asia\lastfm_asia_features.json", "r") as f:
    data = json.load(f)

ids = list(data.keys())

all_features = set()
for features in data.values():
    all_features.update(features)

feature_to_idx = {feature: idx for idx, feature in enumerate(all_features)}
row_indices = []
col_indices = []
for row_idx, (id_, features) in enumerate(data.items()):
    for feature in features:
        col_idx = feature_to_idx[feature]
        row_indices.append(row_idx)
        col_indices.append(col_idx)

# 假设 ids、row_indices、col_indices、all_features、edges 已经定义
data_values = [1] * len(row_indices)
n_row = len(ids)
n_col = len(all_features)

# 构造稀疏矩阵并转为稠密特征矩阵
X = coo_matrix((data_values, (row_indices, col_indices)), shape=(n_row, n_col))
features = X.toarray()

# ---------------- 目标加载 ----------------
target_path = os.path.join("data", "raw", "lasftm_asia", "lastfm_asia_target.csv")
targets = pd.read_csv(target_path)["target"].to_numpy()

# ---------------- 图数据处理 ----------------
print("正在执行图数据预处理...")

sample_size = 100
# 初始化图数据处理器
manager = GraphProcessingManager("lasftm_asia")

# 执行参数网格搜索
manager.grid_search(edges, features, targets, sample_size=sample_size, overwrite=True)

# ---------------- 任务完成标记 ----------------
done_file = os.path.join(manager.storage_manager.intermediate_dir, "done.txt")
os.makedirs(os.path.dirname(done_file), exist_ok=True)
with open(done_file, "w", encoding="utf-8") as f:
    f.write(datetime.datetime.now().strftime("%Y%m%d_%H%M"))

print(f"[Done] 数据处理完成，已写入标记：{done_file}")
