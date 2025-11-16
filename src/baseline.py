"""
Author: Pumpkin🎃
Date:2025-11-10
Description: 对比实验
"""

import json
import os
import time
from typing import Dict, Tuple, List, Union

import numpy as np
import pandas as pd
import networkx as nx

from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

from src.processor import feature_process, edge_process

from cdlib import algorithms


def community2nodelabels(
    communities: List[List[int]], all_node_ids: List[int]
) -> np.ndarray:
    """
    将社区列表转换为 cluster_labels 数组。

    Args:
        communities (List[List[int]]): 社区列表，每个子列表包含一个社区的节点ID。
        all_node_ids (List[int]): 图中所有节点的ID列表。这个列表的顺序决定了输出数组的索引顺序。
                                   通常，这可以通过 sorted(graph.nodes()) 获得。

    Returns:
        Union[np.ndarray, List[int]]: cluster_labels 数组。数组的索引对应 `all_node_ids` 中的节点顺序，
                                      数组的值是该节点的社区ID。如果节点ID不在社区列表中，其值为 -1。
                                      默认返回 NumPy 数组，若需返回列表，可设置 return_as_list=True。
    """
    # 1. 首先创建一个节点到社区ID的映射字典，这是高效查找的关键
    node_to_community = {}
    for community_id, community_nodes in enumerate(communities):
        for node_id in community_nodes:
            node_to_community[node_id] = community_id

    # 2. 创建 cluster_labels 数组
    # 遍历所有节点ID，并根据映射字典为每个节点分配社区ID
    # 如果节点不在任何社区中，则分配 -1
    cluster_labels = [node_to_community.get(node_id, -1) for node_id in all_node_ids]
    # 3. 转换为 NumPy 数组并返回（推荐，便于后续处理）
    return np.array(cluster_labels)


def load_graph_and_features(
    edges_filepath: str,
    features_filepath: str,
    feature_file_type: str = "csv",  # 新增参数：指定特征文件类型，"csv"或"json"
) -> Tuple[nx.Graph, Dict]:
    """
    从边文件和特征文件（CSV/JSON）加载数据，转换为 networkx.Graph 和节点属性字典。

    Args:
        edges_filepath (str): .edges 文件路径（CSV格式，每行两个节点ID）。
        features_filepath (str): .features 文件路径（CSV或JSON格式）。
        feature_file_type (str): 特征文件类型，可选 "csv" 或 "json"，默认 "csv"。

    Returns:
        tuple: (G, node_attributes)
            - G (networkx.Graph): 无向图对象。
            - node_attributes (dict): 节点属性字典，格式 {node_id: {'feat_xxx': 0/1, ...}}。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 特征文件类型无效。
        json.JSONDecodeError: JSON文件格式错误。
    """
    # --- 1. 检查文件是否存在 ---
    for filepath in [edges_filepath, features_filepath]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件未找到: {filepath}")

    # --- 2. 加载并构建图（与之前逻辑一致） ---
    print(f"正在加载边文件: {edges_filepath}")
    edges_df = pd.read_csv(edges_filepath, header=None, names=["source", "target"])
    G = nx.from_pandas_edgelist(edges_df, "source", "target")
    print(f"图构建完成：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")

    # --- 3. 加载并构建节点属性字典（根据文件类型处理） ---
    print(f"正在加载特征文件: {features_filepath}（类型：{feature_file_type}）")
    if feature_file_type == "csv":
        # 原有CSV格式处理逻辑
        features_df = pd.read_csv(features_filepath, header=None)
        node_attributes = {}
        for idx, row in features_df.iterrows():
            node_id = idx  # 假设节点ID=行号（0-based），需调整则改为 idx+1（1-based）
            node_attributes[node_id] = {f"attr_{i}": val for i, val in enumerate(row)}
    elif feature_file_type == "json":
        # 新增JSON格式处理逻辑
        with open(features_filepath, "r", encoding="utf-8") as f:
            feat_dict = json.load(f)  # 读取为 {节点id: [拥有的特征id列表]}

        # 第一步：获取所有特征ID（用于补全"无特征"的属性为0）
        all_feature_ids = set()
        for feat_list in feat_dict.values():
            all_feature_ids.update(feat_list)
        all_feature_ids = sorted(list(all_feature_ids))  # 排序保证属性顺序一致
        print(f"共检测到 {len(all_feature_ids)} 个不同特征")

        # 第二步：构建属性字典（有特征=1，无特征=0）
        node_attributes = {}
        for node_id_str, owned_feats in feat_dict.items():
            # 节点ID可能是字符串（JSON键默认字符串），转换为整数（与边文件节点ID类型一致）
            node_id = int(node_id_str)
            # 为当前节点初始化所有特征为0
            attrs = {f"feat_{fid}": 0 for fid in all_feature_ids}
            # 对拥有的特征，设为1
            for fid in owned_feats:
                attrs[f"feat_{fid}"] = 1
            node_attributes[node_id] = attrs
    else:
        raise ValueError(
            f"无效的特征文件类型：{feature_file_type}，仅支持 'csv' 或 'json'"
        )

    print(f"特征加载完成：为 {len(node_attributes)} 个节点分配属性")

    # --- 4. 验证并清理数据（确保图节点都有属性） ---
    graph_nodes = set(G.nodes())
    attr_nodes = set(node_attributes.keys())
    nodes_without_attrs = graph_nodes - attr_nodes
    nodes_without_graph = attr_nodes - graph_nodes
    if nodes_without_attrs:
        print(f"警告：{len(nodes_without_attrs)} 个图节点无对应特征，将移除")
        G.remove_nodes_from(nodes_without_attrs)
    if nodes_without_graph:
        print(f"警告：{len(nodes_without_graph)} 个特征节点不在图中，将忽略")

    print(f"最终图：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")
    return G, node_attributes


def spectral_clustering(X, k):
    if isinstance(X, csr_matrix):
        X = X.toarray()
    similarity_matrix = X
    # 计算度矩阵 D
    degree_matrix = np.sum(similarity_matrix, axis=1)
    D = np.diag(degree_matrix)

    # 计算规范化拉普拉斯矩阵 L_sym
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix))
    L = D - similarity_matrix
    L_norm = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)
    L_norm = np.nan_to_num(L_norm, nan=0.0, posinf=1e10, neginf=-1e10)
    # 特征值分解
    eigvals, eigvecs = eigh(L_norm)

    # 选择最小的 k 个特征值对应的特征向量
    embedding = eigvecs[:, :k]

    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embedding)

    return labels


def nmf_clustering(X, k):
    if isinstance(X, csr_matrix):
        X = X.toarray()
    # 初始化 NMF 模型
    nmf = NMF(n_components=k, init="random", random_state=42)

    # 对数据进行非负矩阵分解
    W = nmf.fit_transform(X)

    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(W)

    return labels


def baselineExperiment(dataname):
    result = {}
    if "llf" in dataname:
        features_format = "csv"
    else:
        features_format = "json"

    features_file = f"data/graphs/{dataname}.features"

    G, node_attrs = load_graph_and_features(
        edges_filepath=f"data/graphs/{dataname}.edges",
        features_filepath=features_file,
        feature_file_type=features_format,  # 关键：指定为JSON类型
    )
    targets = pd.read_csv(f"data/graphs/{dataname}.targets", header=None)
    time_start = time.time()
    comminities = algorithms.ilouvain(G, node_attrs)

    time_end = time.time()
    print(f"ilouvain 运行时间: {time_end - time_start} 秒")
    result["ilouvain"] = {
        "time": time_end - time_start,
    }
    cluster_labels = community2nodelabels(
        comminities.communities, sorted(list(G.nodes()))
    )
    if os.path.exists(f"results/baseline/{dataname}.baseline"):
        baseline_metrics = pd.read_csv(f"results/baseline/{dataname}.baseline", header=None)
        baseline_metrics['ilouvain'] = cluster_labels
        with open(f"results/baseline/{dataname}.time", "a", encoding="utf-8") as f:
            f.write(f"ilouvain_time: {time_end - time_start} seconds\n")
            
    else:
        print("baseline 指标文件不存在")



    # time_start = time.time()
    # comminities = algorithms.eva(G, node_attrs)
    # time_end = time.time()
    # print(f"eva 运行时间: {time_end - time_start} 秒")
    # result["eva"] = {
    #     "time": time_end - time_start,
    # }
    # cluster_labels = community2nodelabels(
    #     comminities.communities, sorted(list(G.nodes()))
    # )
    # eva = Evaluator(cluster_labels, targets.values[:, 1])
    # result["eva"].update(eva.get_all_metrics())
    
    return 

if __name__ == "__main__":
    print(baselineExperiment("llf_friendship"))
