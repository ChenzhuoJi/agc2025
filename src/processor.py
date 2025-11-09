"""
Author: Pumpkin🎃
Date:2025-11-07
Description: 图处理模块
"""

import json
import time
import warnings
from typing import List, Union
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances


def is_consecutive(lst: List[int]) -> bool:
    if not lst:  # 如果列表为空
        return False

    # 排序列表
    lst = list(set(lst))
    lst_sorted = sorted(lst)

    # 检查相邻元素差是否为 1
    for i in range(1, len(lst_sorted)):
        if lst_sorted[i] - lst_sorted[i - 1] != 1:
            return False

    return True


def is_sparse_based_on_density(matrix: np.ndarray, threshold: float = 0.9) -> bool:
    # 判断矩阵是否为稀疏
    # 计算零元素的比例
    zero_count = np.sum(matrix == 0)
    total_elements = matrix.size
    zero_density = zero_count / total_elements
    return zero_density > threshold


def edge_process(
    dataname: str, undirected: bool = True, sparsity_threshold: float = 0.9
) -> Union[np.ndarray, sp.csr_matrix]:
    edges_file = f"data/graphs/{dataname}.edges"
    edges = pd.read_csv(edges_file, header=None)
    edges.columns = ["id1", "id2"]

    edges = edges.to_numpy()

    n = edges.max() + 1
    adj = np.zeros((n, n))

    for u, v in edges:
        adj[u, v] = 1
        if undirected:
            adj[v, u] = 1

    # 根据零元素的密度判断是否需要转换为稀疏矩阵
    if is_sparse_based_on_density(adj, sparsity_threshold):
        adj = sp.csr_matrix(adj)  # 转换为稀疏矩阵（CSR格式）
        print("邻接矩阵转换为稀疏矩阵")

    return adj


def feature_process(
    dataname: str, sigma: float = 0.5, sparsity_threshold: float = 0.9
) -> np.ndarray:
    features_file = f"data/graphs/{dataname}.features"

    with open(features_file, "r") as f:
        features_data = json.load(f)
        nodes = list(features_data.keys())

    num_nodes = len(nodes)

    # 获取所有特征的最大索引值，确定特征总数
    all_features = sorted([f for features in features_data.values() for f in features])
    if not is_consecutive(all_features):
        warnings.warn("特征索引不是连续的整数，可能会导致错误")

    num_features = max(all_features) + 1  # 因为特征从 0 开始索引

    # 创建稀疏特征矩阵
    features = np.zeros((num_nodes, num_features))
    for i, node in enumerate(nodes):
        for feature in features_data[node]:
            features[i, feature] = 1

    # 根据零元素的密度判断是否需要转换为稀疏矩阵
    if is_sparse_based_on_density(features, sparsity_threshold):
        features = sp.csr_matrix(features)  # 转换为稀疏矩阵（CSR格式）
        print("特征矩阵转换为稀疏矩阵")

    # 计算特征相似度矩阵
    if sp.issparse(features):
        features_sq = features.power(2).sum(axis=1).A1
        dot_product = features @ features.T
        dists_sq = (
            features_sq[:, None] + features_sq[None, :] - 2 * dot_product.toarray()
        )
    else:
        dists_sq = pairwise_distances(features, metric="sqeuclidean")

    similarity_matrix = np.exp(-sigma * dists_sq)
    # 检查相似度矩阵中的 NaN 和 inf 值
    if np.any(np.isnan(similarity_matrix)) or np.any(np.isinf(similarity_matrix)):
        warnings.warn("相似度矩阵中包含 NaN 或 inf 值！")

    return similarity_matrix


def high_order(
    term: Union[np.ndarray, sp.csr_matrix], order: int = 2, decay: float = 0.5
) -> np.ndarray:
    if sp.issparse(term):  # 如果是稀疏矩阵
        ho_matrix = sp.csr_matrix(term.shape, dtype=np.float32)  # 初始化高阶矩阵
        matrix_power = term.copy()  # 当前矩阵的幂，初始为 term
        factorial = 1.0
        for i in range(1, order + 1):
            factorial *= i
            ho_matrix += matrix_power.multiply(decay**i / factorial)  # 累加高阶项
            matrix_power = matrix_power @ term  # 更新矩阵的幂
    else:  # 如果是稠密矩阵
        ho_matrix = np.zeros_like(term, dtype=np.float32)  # 初始化高阶矩阵
        matrix_power = term.copy()  # 当前矩阵的幂，初始为 term
        factorial = 1.0
        for i in range(1, order + 1):
            factorial *= i
            ho_matrix += matrix_power * (decay**i / factorial)  # 累加高阶项
            matrix_power = matrix_power @ term  # 更新矩阵的幂
    if sp.issparse(ho_matrix):
        ho_matrix = ho_matrix.toarray()
    return ho_matrix


if __name__ == "__main__":
    t = time.time()
    edge_process("citeseer")
    print(f"邻接矩阵处理耗时: {time.time() - t}")
    t = time.time()
    feature_process("citeseer")
    print(f"特征矩阵处理耗时: {time.time() - t}")
    t = time.time()
    high_order(edge_process("citeseer"))
    print(f"高阶传播矩阵处理耗时: {time.time() - t}")
