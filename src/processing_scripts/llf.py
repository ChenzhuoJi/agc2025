import os

from data.processor import feture_preprocess, edge_preprocess, high_order, get_data_path

import pandas as pd
import numpy as np

data_dir_name = "LLF/"
raw_data_dir, processed_data_dir, interim_data_dir = get_data_path(data_dir_name)

ml_edges = pd.read_csv(
    raw_data_dir + "Lazega-Law-Firm_multiplex.edges", sep=" ", header=None
)
ml_edges.columns = ["layer", "node1", "node2", "weight"]
ml_edges.to_csv(os.path.join(processed_data_dir, "edges.csv"), index=False)

all_nodes = set(ml_edges["node1"].unique()) | set(ml_edges["node2"].unique())
all_nodes = sorted(list(all_nodes))
n_nodes = len(all_nodes)

node_to_index = {node: idx for idx, node in enumerate(all_nodes)}
index_to_node = {idx: node for node, idx in node_to_index.items()}

adj_matrices = {}
layers = sorted(ml_edges["layer"].unique())
for layer_id in layers:
    # 提取当前层的边
    layer_edges = ml_edges[ml_edges["layer"] == layer_id]

    # 初始化邻接矩阵
    adj_matrix = np.zeros((n_nodes, n_nodes))

    # 填充边信息
    for _, edge in layer_edges.iterrows():
        node1_idx = node_to_index[edge["node1"]]
        node2_idx = node_to_index[edge["node2"]]
        adj_matrix[node1_idx, node2_idx] = edge["weight"]
        adj_matrix[node2_idx, node1_idx] = edge["weight"]  # 无向图对称处理

    # 计算高阶邻接矩阵
    ho_matrix = high_order(adj_matrix)

    # 保存邻接矩阵
    np.save(interim_data_dir + f"adj_matrix{layer_id}.npy", adj_matrix)
    np.save(interim_data_dir + f"ho_matrix{layer_id}.npy", ho_matrix)


# ----------------feature------------------
features = pd.read_csv(
    os.path.join(raw_data_dir, "Lazega-Law-Firm_nodes.txt"),
    sep="\\s+",  # 使用正则表达式匹配任意空白字符
    header=0,
)
features.to_csv(os.path.join(processed_data_dir, "features.csv"), index=False)
similarity_matrix = feture_preprocess(features)
np.save(interim_data_dir + "similarity_matrix.npy", similarity_matrix)
