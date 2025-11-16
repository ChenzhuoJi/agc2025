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

import json
import os


def detect_features_format(file_path: str) -> str:
    """
    检测.features文件的格式（JSON 或 CSV）

    Args:
        file_path: 特征文件的完整路径

    Returns:
        str: 检测结果，"json" 或 "csv"

    Raises:
        ValueError: 无法识别的文件格式
    """
    # 读取文件前3行采样，避免读取大文件时性能问题
    with open(file_path, "r", encoding="utf-8") as f:
        sample_lines = []
        for _ in range(3):
            line = f.readline().strip()
            if line:
                sample_lines.append(line)
            else:
                break  # 若提前读到空行，停止采样

    # 1. 检测是否为JSON格式（首行{、末行}，且能成功解析）
    if (
        sample_lines
        and sample_lines[0].startswith("{")
        and sample_lines[-1].endswith("}")
    ):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json.load(f)  # 验证是否为合法JSON
            return "json"
        except json.JSONDecodeError:
            pass  # 解析失败，排除JSON格式

    # 2. 检测是否为CSV格式（包含常见分隔符）
    common_delimiters = [",", "\t", ";"]  # 支持逗号、制表符、分号分隔
    for delimiter in common_delimiters:
        if any(delimiter in line for line in sample_lines):
            return "csv"

    # 3. 无法识别的格式
    raise ValueError(
        f"不支持的.features文件格式！\n"
        f"文件路径：{file_path}\n"
        f'支持格式：1. JSON格式（{{"节点id": [特征id列表]}}）；2. CSV格式（行=节点，列=特征）'
    )


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
    """处理图数据集的边信息，并构建邻接矩阵表示。

    该函数读取指定数据集的边文件，构建图的邻接矩阵，并根据稀疏性判断是否转换为
    稀疏矩阵表示，以优化内存使用和计算效率。

    Args:
        dataname (str): 数据集名称，用于确定边文件路径
        undirected (bool): 图是否为无向图，如果为True则邻接矩阵将是对称的，默认为True
        sparsity_threshold (float): 判断是否使用稀疏矩阵表示的阈值，当矩阵稀疏度超过此值时使用稀疏表示，默认为0.9

    Returns:
        Union[np.ndarray, sp.csr_matrix]: 构建的邻接矩阵，可以是NumPy稠密数组或SciPy CSR格式的稀疏矩阵
    """
    # 构建边文件路径
    edges_file = f"stgraphs/{dataname}.edges"

    # 读取边数据文件
    edges = pd.read_csv(edges_file, header=None)
    edges.columns = ["id1", "id2"]  # 为边数据添加列名

    # 转换为NumPy数组格式，便于后续处理
    edges = edges.to_numpy()

    # 确定图中节点数量（假设节点ID连续且从0开始）
    n = edges.max() + 1

    # 初始化邻接矩阵（全零矩阵）
    adj = np.zeros((n, n))

    # 填充邻接矩阵
    for u, v in edges:
        adj[u, v] = 1  # 设置边(u,v)存在
        if undirected:
            adj[v, u] = 1  # 无向图中，边(v,u)也存在

    # 根据零元素的密度判断是否需要转换为稀疏矩阵，以节省内存
    if is_sparse_based_on_density(adj, sparsity_threshold):
        adj = sp.csr_matrix(adj)  # 转换为压缩稀疏行(CSR)格式

    return adj  # 返回构建的邻接矩阵


def feature_process(
    dataname: str, sigma: float = 0.5, sparsity_threshold: float = 0.9
) -> np.ndarray:
    """处理图数据集的节点特征，并计算基于高斯核的特征相似度矩阵。

    该函数读取指定数据集的特征文件，构建节点-特征矩阵，根据稀疏性判断是否转换为稀疏表示，
    并最终计算基于高斯核的节点间特征相似度矩阵，用于后续图表示学习。

    Args:
        dataname (str): 数据集名称，用于确定特征文件路径
        sigma (float): 高斯核函数的带宽参数，控制相似度衰减速度，默认为0.5
        sparsity_threshold (float): 判断是否使用稀疏矩阵表示的阈值，当矩阵稀疏度超过此值时使用稀疏表示，默认为0.9

    Returns:
        np.ndarray: 节点间的特征相似度矩阵，形状为[节点数量, 节点数量]
    """
    # 构建特征文件路径
    features_file = f"stgraphs/{dataname}.features"

    # 读取特征文件数据
    with open(features_file, "r") as f:
        features_data = json.load(f)  # 加载JSON格式的特征数据
        nodes = list(features_data.keys())  # 获取所有节点的列表

    num_nodes = len(nodes)  # 计算节点数量

    # 获取所有特征的最大索引值，确定特征总数
    all_features = sorted([f for features in features_data.values() for f in features])
    if not is_consecutive(all_features):
        warnings.warn("特征索引不是连续的整数，但是依旧可以计算。")

    num_features = (
        max(all_features) + 1
    )  # 因为特征从 0 开始索引，所以特征总数为最大索引+1

    # 创建稀疏特征矩阵（初始为稠密矩阵）
    features = np.zeros((num_nodes, num_features))
    for i, node in enumerate(nodes):
        for feature in features_data[node]:
            features[i, feature] = 1  # 特征存在则标记为1，不存在为0

    # 根据零元素的密度判断是否需要转换为稀疏矩阵，以节省内存和加速计算
    if is_sparse_based_on_density(features, sparsity_threshold):
        features = sp.csr_matrix(features)  # 转换为压缩稀疏行(CSR)格式

    # 计算特征相似度矩阵（使用高斯核函数）
    if sp.issparse(features):
        # 稀疏矩阵计算策略
        features_sq = features.power(2).sum(axis=1).A1  # 计算每个节点特征的平方和
        dot_product = features @ features.T  # 计算节点间特征的点积
        # 使用点积计算欧氏距离的平方
        dists_sq = (
            features_sq[:, None] + features_sq[None, :] - 2 * dot_product.toarray()
        )
    else:
        # 稠密矩阵直接计算欧氏距离的平方
        dists_sq = pairwise_distances(features, metric="sqeuclidean")

    # 应用高斯核函数将距离转换为相似度
    similarity_matrix = np.exp(-sigma * dists_sq)

    # 检查相似度矩阵中的 NaN 和 inf 值，避免后续计算错误
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
    edge_process("cora")
    print(f"邻接矩阵处理耗时: {time.time() - t}")
    t = time.time()
    la = feature_process("cora")
    la = high_order(la)

    la /= np.max(la)
    print(f"特征矩阵处理耗时: {time.time() - t}")
    t = time.time()
    ls = high_order(edge_process("cora"))
    ls /= np.max(ls)
    print(f"高阶传播矩阵处理耗时: {time.time() - t}")