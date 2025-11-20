"""
Author: Pumpkin🎃
Date:2025-11-07
Description: 图处理模块
"""

import json
import warnings
from typing import List, Union
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel
import json


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


def feature_process(featmat, kernel, sparse=True, gamma=0.5):
    """
    计算特征矩阵的相似度核矩阵。
    同时支持稀疏矩阵和稠密矩阵，并针对不同类型进行了优化。

    Args:
        featmat: 输入特征矩阵 (scipy.sparse.csr_matrix 或 numpy.ndarray)。
        kernel: 相似度核类型 ('linear', 'cosine', 'jaccard', 'rbf')。
        sparse: 是否将输入视为稀疏矩阵。
        gamma: RBF核的带宽参数。

    Returns:
        numpy.ndarray: 稠密的相似度矩阵。
    """
    X = featmat.tocsr() if sparse else featmat

    similarity = None

    if kernel == "linear":
        if sparse:
            similarity = (X @ X.T).tocsr()
        else:
            similarity = X @ X.T

    elif kernel == "cosine":
        X_norm = normalize(X, norm="l2", axis=1)
        if sparse:
            similarity = (X_norm @ X_norm.T).tocsr()
        else:
            similarity = X_norm @ X_norm.T

    elif kernel == "jaccard":
        # Jaccard核的计算天然适合二值化特征
        if sparse:
            inter = X @ X.T
            row_sums = np.array(X.sum(axis=1)).flatten()
            unions = row_sums[:, np.newaxis] + row_sums[np.newaxis, :] - inter.toarray()
            similarity = inter.toarray() / (unions + 1e-12)
        else:
            inter = X @ X.T
            row_sums = np.array(X.sum(axis=1)).flatten()
            unions = row_sums[:, np.newaxis] + row_sums[np.newaxis, :] - inter
            similarity = inter / (unions + 1e-12)

    elif kernel == "rbf":
        if sparse:
            # 稀疏二值矩阵的优化实现
            popcount = np.array(X.sum(axis=1)).flatten()
            intersection = (X @ X.T).toarray()
            dist2 = popcount[:, np.newaxis] + popcount[np.newaxis, :] - 2 * intersection
            similarity = np.exp(-gamma * dist2)
        else:
            # 稠密矩阵使用sklearn的高效实现
            similarity = rbf_kernel(X, gamma=gamma)

    else:
        raise ValueError(f"不支持的核函数类型: {kernel}")

    # 确保最终输出是稠密的numpy数组
    if sp.issparse(similarity):
        similarity = similarity.toarray()

    assert isinstance(similarity, np.ndarray), "相似度矩阵必须为NumPy数组"
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
