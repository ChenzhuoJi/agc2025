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
from sklearn.preprocessing import normalize
import json
import os


def featjson2sparse(features_file):
    """直接将特征文件加载为稀疏矩阵

    Args:
        features_file (str): 特征文件路径

    Returns:
        scipy.sparse.csr_matrix: 稀疏特征矩阵
    """
    # 读取特征文件数据
    with open(features_file, "r") as f:
        features_data = json.load(f)

    nodes = list(features_data.keys())
    num_nodes = len(nodes)

    # 收集所有非零元素的行索引、列索引和值
    row_indices = []
    col_indices = []
    data = []

    # 遍历每个节点及其特征
    for i, node in enumerate(nodes):
        for feature_idx in features_data[node]:
            row_indices.append(i)
            col_indices.append(feature_idx)
            data.append(1)  # 特征存在则为1

    # 确定矩阵的形状
    num_features = max(col_indices) + 1 if col_indices else 0

    # 创建稀疏矩阵
    features_sparse = sp.csr_matrix(
        (data, (row_indices, col_indices)), shape=(num_nodes, num_features)
    )

    return features_sparse


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


def feature_process(featmat, kernel):
    X = featmat.tocsr()
    if kernel == "linear":
        # K = X X^T, 可为稀疏矩阵
        similarity = (X @ X.T).tocsr()
    if kernel == "cosine":
        X_norm = normalize(X, norm="l2", axis=1)
        similarity = (X_norm @ X_norm.T).tocsr()
    if kernel == "jaccard":
        # intersection = dot product for 0/1
        inter = X @ X.T  # sparse
        inter = inter.tocsr()

        row_sums = np.array(X.sum(axis=1)).reshape(-1)
        # union = A + B - intersection
        # 注意：这里的广播不会稠密化 intersection（它本身是稀疏）
        unions = row_sums[:, None] + row_sums[None, :] - inter.toarray()

        # Jaccard 必须返回 dense（unions 会 dense）
        similarity = inter.toarray() / (unions + 1e-12)
    if kernel == "rbf":
        gamma = 0.5
        """优化后的稀疏二值矩阵RBF核计算"""
        # 每行1的个数（因为x^2 = x对于二值数据）
        popcount = np.array(X.sum(axis=1)).flatten()

        # 交集大小（点积）
        intersection = (X @ X.T).toarray()

        # 平方距离
        dist2 = popcount[:, None] + popcount[None, :] - 2 * intersection

        # RBF核
        similarity = np.exp(-gamma * dist2)
    if sp.issparse(similarity):
        similarity = similarity.toarray()
    assert type(similarity) == np.ndarray, "相似度矩阵必须为NumPy数组"
    return similarity


def high_order_old(
    term: Union[sp.csr_matrix, np.ndarray], order: int = 2, decay: float = 0.5
) -> Union[sp.csr_matrix, np.ndarray]:
    """
    计算高阶矩阵和： sum_{i=1..order} (decay^i / i!) * (term)^i
    支持：
        - 稀疏 CSR 矩阵
        - Dense ndarray
    输出保持与输入相同类型：
        输入 CSR → 输出 CSR
        输入 ndarray → 输出 ndarray
    """
    is_sparse = sp.issparse(term)
    # ---------- 初始化 ----------
    if is_sparse:
        term = term.tocsr()
        ho_matrix = sp.csr_matrix(term.shape, dtype=np.float32)
        matrix_power = term.copy()
    else:
        ho_matrix = np.zeros_like(term, dtype=np.float32)
        matrix_power = term.copy()
    factorial = 1.0
    # ---------- 主循环 ----------
    for i in range(1, order + 1):
        factorial *= i
        coeff = (decay**i) / factorial
        if is_sparse:
            ho_matrix += matrix_power.multiply(coeff)
            matrix_power = matrix_power @ term  # CSR @ CSR仍是CSR
        else:
            ho_matrix += matrix_power * coeff
            matrix_power = matrix_power @ term
    # ---------- 稀疏密度检查 ----------
    if is_sparse:
        density = ho_matrix.nnz / (ho_matrix.shape[0] * ho_matrix.shape[1])
        if density > 0.1:
            warnings.warn(f"高阶矩阵密度过高: {density:.4%}")
    return ho_matrix

def high_order(
    term: Union[sp.csr_matrix, np.ndarray], order: int = 2, decay: float = 0.5
) -> Union[sp.csr_matrix, np.ndarray]:
    """
    计算高阶矩阵和： sum_{i=1..order} (decay^i / i!) * (term)^i
    支持：
        - 稀疏 CSR 矩阵
        - Dense ndarray
    输出保持与输入相同类型：
        输入 CSR → 输出 CSR
        输入 ndarray → 输出 ndarray
    """
    is_sparse = sp.issparse(term)
    dtype = term.dtype if not is_sparse else term.dtype
    # ---------- 初始化 ----------
    if is_sparse:
        term = term.tocsr()
        ho_matrix = sp.csr_matrix(term.shape, dtype=dtype)
        matrix_power = term.copy()
    else:
        ho_matrix = np.zeros_like(term, dtype=dtype)
        matrix_power = term.copy()
    factorial = 1.0

    # ---------- 主循环 ----------
    for i in range(1, order + 1):
        # 计算系数
        factorial *= i
        if i == 1:
            decay_pow = 1
        else:
            decay_pow *= decay
        coeff = (decay_pow) / factorial

        # 更新高阶矩阵
        if is_sparse:
            ho_matrix += matrix_power.multiply(coeff)
        else:
            ho_matrix += matrix_power * coeff

        # 更新下一个矩阵幂
        if i < order:
            matrix_power = matrix_power @ term

    # 稀疏密度检查 
    if is_sparse:
        density = ho_matrix.nnz / (ho_matrix.shape[0] * ho_matrix.shape[1])
        if density > 0.1:
            warnings.warn(f"高阶矩阵密度过高: {density:.4%}")
    return ho_matrix

if __name__ == "__main__":
    pass
