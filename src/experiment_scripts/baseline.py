import os
import json

from networkx import adjacency_matrix
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from scipy.sparse import coo_matrix

from src.evaluator import Evaluator
from src.processor import edge_preprocess
# # baseline
# data = pd.read_csv("data/raw/LLF/Lazega-Law-Firm_nodes.txt", sep=r"\s+", header=0)
# print(data.head(5))
# X = data.drop(columns=["nodeOffice","nodeID"])
# y = data["nodeOffice"]
# print(X.columns)
# cluster_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
# eva = Evaluator(cluster_labels, y)
# eva.print_metrics()

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
print('!')
# ---------------- 目标加载 ----------------
target_path = os.path.join("data", "raw", "lasftm_asia", "lastfm_asia_target.csv")
targets = pd.read_csv(target_path)["target"].to_numpy()

sample_size = 500  # 你自己设定

# 随机抽取 sample_size 个索引
indices = np.random.choice(features.shape[0], size=sample_size, replace=False)

# 按索引抽取样本
features_sampled = features[indices]
targets_sampled = targets[indices]

r = len(np.unique(targets_sampled))

kmeans = KMeans(n_clusters=r, random_state=42)
cluter_labels = kmeans.fit_predict(features_sampled)

from src.evaluator import Evaluator

eva = Evaluator(cluter_labels, targets_sampled)
eva.print_metrics()

adjacency_matrix,_ = edge_preprocess(edges)
adj_sampled = adjacency_matrix[indices, :][:, indices]

print(adj_sampled.shape)
nmf = NMF(n_components=r,random_state=42)
U = nmf.fit_transform(adj_sampled)
cluster_labels = np.argmax(U, axis=1)

eva = Evaluator(cluster_labels, targets_sampled)
eva.print_metrics()
