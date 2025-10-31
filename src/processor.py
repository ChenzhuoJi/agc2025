import numpy as np
from sklearn.metrics import pairwise_distances
from src.helpers import dataStorageManager, paramsManager
import os

import scipy.sparse as sp
import time


def edge_preprocess(edges, undirected=True):
    """edges: [
    [node_i1, node_j1],
    [node_i2, node_j2],
    ...
    ] -> adjacency matrix"""
    nodes = np.unique(edges)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    adjacency_matrix = np.zeros((n, n), dtype=float)
    for u, v in edges:
        i, j = node_to_index[u], node_to_index[v]
        adjacency_matrix[i, j] = 1
        if undirected:
            adjacency_matrix[j, i] = 1
    return adjacency_matrix, node_to_index


def high_order(adjacency_matrix, p=2, theta=0.5, sparse=True):
    if sparse:
        if not sp.issparse(adjacency_matrix):
            adjacency_matrix = sp.csr_matrix(adjacency_matrix)
        ho_matrix = sp.csr_matrix(adjacency_matrix.shape, dtype=np.float32)
        A_power = adjacency_matrix.copy()
        factorial = 1.0
        for i in range(1, p + 1):
            factorial *= i
            ho_matrix += A_power.multiply(theta**i / factorial)
            A_power = A_power @ adjacency_matrix
        return ho_matrix
    else:
        ho_matrix = np.zeros_like(adjacency_matrix, dtype=np.float32)
        A_power = adjacency_matrix.copy()
        factorial = 1.0
        for i in range(1, p + 1):
            factorial *= i
            ho_matrix += A_power * (theta**i / factorial)
            A_power = A_power @ adjacency_matrix
        return ho_matrix


def communicability_matrix(adjacency_matrix, similarity_matrix, p, theta):
    pass


def feature_preprocess(features):
    dists_sq = pairwise_distances(features, metric="sqeuclidean")
    similarity_matrix = np.exp(-0.5 * dists_sq)
    return similarity_matrix


import os
import time
import numpy as np
import scipy.sparse as sp


class GraphPreprocessor:
    """单次图数据处理器"""

    def __init__(self, p=2, theta=0.5):
        self.p = p
        self.theta = theta

    def process(self, edges, features, targets, sample_size=None):
        """
        对输入图数据执行特征与高阶矩阵处理
        """
        adj, node_to_index = edge_preprocess(edges)
        n_nodes = features.shape[0]
        res = {
            "parameters": {"p": self.p, "theta": self.theta},
            "node_to_index": node_to_index,
            "sampled": False,
        }

        # --- 节点采样 ---
        if sample_size and sample_size < n_nodes:
            sample_nodes = np.random.choice(n_nodes, sample_size, replace=False)
            res["sampled"] = True
            res["sample_nodes"] = sample_nodes
            features_used = features[sample_nodes, :]
            targets_used = targets[sample_nodes]
        else:
            sample_nodes = None
            features_used = features
            targets_used = targets

        # --- 计算相似度矩阵 ---
        start = time.time()
        similarity = feature_preprocess(features_used)
        res["similarity_matrix"] = similarity
        print(f"[Similarity matrix] computed in {time.time() - start:.4f}s")

        # --- 计算高阶矩阵 ---
        start = time.time()
        ho_matrix = high_order(adj, self.p, self.theta)
        if sample_nodes is not None:
            if sp.issparse(ho_matrix):
                ho_matrix = ho_matrix[np.ix_(sample_nodes, sample_nodes)]
            else:
                ho_matrix = ho_matrix[np.ix_(sample_nodes, sample_nodes)]
        res["high_order_matrix"] = ho_matrix
        print(f"[High-order matrix] computed in {time.time() - start:.4f}s")
        res["targets"] = targets_used
        self.res = res
        return res


class GraphProcessingManager:
    """图数据处理任务管理器（含存储、网格搜索等功能）"""

    def __init__(self, dataname):
        self.dataname = dataname
        self.raw_dir = os.path.join("data/raw", dataname)
        self.inter_dir = os.path.join("data/intermediate", dataname)

        self.storage_manager = dataStorageManager(dataname)

    def run(
        self, edges, features, targets, p=2, theta=0.5, sample_size=1000, overwrite=False
    ):
        """执行单组参数的图数据处理流程

        此方法创建GraphPreprocessor实例，并使用指定参数对图数据进行预处理，
        包括计算结构层特征、属性层特征以及融合后的交互层特征，最后可选择保存处理结果。

        参数:
            edges: 图的边数据，表示节点之间的连接关系
            features: 节点的属性特征数据
            targets: 节点的标签/目标值
            p: 高阶结构信息的阶数，默认为2
            theta: 结构信息和属性信息的融合权重参数，默认为0.5
            sample_size: 采样大小，用于处理大规模图数据时的采样策略，默认为1000
            overwrite: 是否覆盖已存在的处理结果文件，默认为False

        返回值:
            dict: 包含预处理结果的字典，包括：
                - similarity_matrix: 属性相似度矩阵
                - high_order_sample: 高阶结构采样矩阵
                - dataname: 数据集名称
                - targets: 节点的标签/目标值
                以及其他预处理过程中生成的中间数据
        """
        # 创建图预处理器实例，传入高阶参数p和融合参数theta
        preprocessor = GraphPreprocessor(p, theta)
        # 执行预处理过程，获取处理结果
        res = preprocessor.process(edges, features, targets, sample_size)
        # 添加数据集名称到结果字典
        res["dataname"] = self.dataname
        # 如果save为True，保存处理结果
        self.storage_manager.save(res, overwrite=overwrite)
        # 返回预处理结果
        return res

    def grid_search(
        self, edges, features, targets, sample_size=1000, overwrite=False,
    ):
        """对 (p, theta) 参数组合进行网格搜索批量处理

        此方法实现了对不同(p, theta)参数组合的自动化遍历处理，通过调用run方法对每一组参数执行完整的数据处理流程。
        支持结果存在性检查，避免重复计算。

        参数:
            edges: 图的边数据，包含节点间连接关系
            features: 节点的特征数据
            targets: 目标变量数据
            sample_size: 样本大小，默认为1000
            overwrite: 是否覆盖已存在的处理结果文件，默认为False

        返回值:
            self: 返回实例本身，支持链式调用
        """
        pm = paramsManager()  # 创建参数管理器实例，用于获取待搜索的参数范围
        for p in pm.p_to_select:  # 遍历所有待选择的p参数值
            for theta in pm.theta_to_select:  # 遍历所有待选择的theta参数值
                self.storage_manager.params = (p, theta, sample_size)  # 设置当前参数组合
                # 调用run方法执行具体的数据处理流程
                self.run(edges, features, targets, p, theta, sample_size, overwrite=overwrite)
        return self  # 返回实例本身，支持链式调用


def compare_high_order(edges, features, targets, p=2, theta=0.5):
    adjacency_matrix, _ = edge_preprocess(edges)
    # 稠密计算
    start = time.time()
    high_order(adjacency_matrix, p=p, theta=theta, sparse=False)
    dense_time = time.time() - start
    print(f"[Dense]   time: {dense_time:.4f} s")
    # 稀疏计算
    start = time.time()
    high_order(adjacency_matrix, p=p, theta=theta, sparse=True)
    sparse_time = time.time() - start
    print(f"[Sparse]  time: {sparse_time:.4f} s")
    # 加速倍数
    speedup = dense_time / sparse_time if sparse_time > 0 else float("inf")
    print(f"Speedup (Dense / Sparse): {speedup:.2f}x")
    return {"dense_time": dense_time, "sparse_time": sparse_time, "speedup": speedup}


if __name__ == "__main__":
    import pandas as pd

    edges = pd.read_csv(r"data\raw\lasftm_asia\lastfm_asia_edges.csv").to_numpy()
    compare_high_order(edges, p=3, theta=2)
