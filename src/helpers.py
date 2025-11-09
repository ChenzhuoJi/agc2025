"""
Author: Pumpkin🎃
Date:2025-10-28
Description:helper functions
    class GraphAnalysis: 图数据探索分析工具类
    function compute_communititude_metrice: 计算社区指标
    function create_mapping: 创建节点索引映射
"""

import os
import json
import joblib
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import networkx as nx
from collections import Counter

import warnings

warnings.filterwarnings("ignore")

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

class GraphAnalysis:
    def __init__(self, adjacency_matrix):
        """
        优化版稀疏图分析

        参数:
        adjacency_matrix: np.array 或 scipy.sparse矩阵, 图的邻接矩阵
        """
        # 转换为稀疏矩阵格式以节省内存
        if sparse.issparse(adjacency_matrix):
            self.adj_matrix = adjacency_matrix
        else:
            self.adj_matrix = sparse.csr_matrix(adjacency_matrix)

        self.n_nodes = self.adj_matrix.shape[0]

        # 对于大图，不立即创建networkx图对象
        self.G = None

    def basic_statistics(self):
        """优化的基本统计信息计算"""
        print("=" * 50)
        print("图的基本统计信息 (优化版)")
        print("=" * 50)

        start_time = time.time()

        # 使用稀疏矩阵操作计算边数
        n_edges = self.adj_matrix.nnz // 2  # 无向图

        # 图密度
        max_possible_edges = self.n_nodes * (self.n_nodes - 1) / 2
        density = n_edges / max_possible_edges if max_possible_edges > 0 else 0

        # 检查是否对称（无向图）
        if sparse.issparse(self.adj_matrix):
            is_symmetric = (self.adj_matrix != self.adj_matrix.T).nnz == 0
        else:
            is_symmetric = np.allclose(self.adj_matrix, self.adj_matrix.T)

        elapsed_time = time.time() - start_time

        print(f"节点数量: {self.n_nodes:,}")
        print(f"边数量: {n_edges:,}")
        print(f"图密度: {density:.6f}")
        print(f"图类型: {'无向图' if is_symmetric else '有向图'}")
        print(f"计算时间: {elapsed_time:.2f}秒")
        print(f"稀疏度: {(1-density)*100:.2f}%")

        return {
            "n_nodes": self.n_nodes,
            "n_edges": n_edges,
            "density": density,
            "is_directed": not is_symmetric,
            "sparsity": (1 - density),
        }

    def degree_analysis(self, sample_size=1000):
        """优化的度分布分析，支持抽样"""
        print("\n" + "=" * 50)
        print("度分布分析 (优化版)")
        print("=" * 50)

        start_time = time.time()

        # 使用稀疏矩阵的快速度计算
        if self._is_undirected():
            degrees = np.array(self.adj_matrix.sum(axis=1)).flatten()
        else:
            in_degrees = np.array(self.adj_matrix.sum(axis=0)).flatten()
            out_degrees = np.array(self.adj_matrix.sum(axis=1)).flatten()
            degrees = in_degrees + out_degrees

        # 基本统计
        degree_stats = {
            "mean": np.mean(degrees),
            "std": np.std(degrees),
            "max": np.max(degrees),
            "min": np.min(degrees),
            "median": np.median(degrees),
        }

        print(f"平均度: {degree_stats['mean']:.2f}")
        print(f"度标准差: {degree_stats['std']:.2f}")
        print(f"最大度: {degree_stats['max']}")
        print(f"最小度: {degree_stats['min']}")
        print(f"度中位数: {degree_stats['median']}")

        # 度分布抽样分析
        if self.n_nodes > sample_size:
            sampled_indices = np.random.choice(self.n_nodes, sample_size, replace=False)
            sampled_degrees = degrees[sampled_indices]
            print(f"\n基于 {sample_size} 个节点的抽样分析:")
            print(f"  抽样平均度: {np.mean(sampled_degrees):.2f}")
            print(f"  抽样度标准差: {np.std(sampled_degrees):.2f}")

        elapsed_time = time.time() - start_time
        print(f"计算时间: {elapsed_time:.2f}秒")

        return degrees, degree_stats

    def _is_undirected(self):
        """检查图是否无向"""
        if sparse.issparse(self.adj_matrix):
            return (self.adj_matrix != self.adj_matrix.T).nnz == 0
        else:
            return np.allclose(self.adj_matrix, self.adj_matrix.T)

    def connected_components_analysis(self, max_components=10):
        """优化的连通分量分析"""
        print("\n" + "=" * 50)
        print("连通分量分析 (优化版)")
        print("=" * 50)

        if not self._is_undirected():
            print("有向图的连通性分析较为复杂，此处省略")
            return None, None, None

        start_time = time.time()

        # 使用scipy的连通分量算法（比自定义BFS快得多）
        n_components, labels = sparse.csgraph.connected_components(
            self.adj_matrix, directed=False, return_labels=True
        )

        # 计算各连通分量大小
        component_sizes = Counter(labels)
        sorted_components = sorted(
            component_sizes.items(), key=lambda x: x[1], reverse=True
        )

        print(f"连通分量数量: {n_components}")
        print(f"最大连通分量大小: {sorted_components[0][1]}")
        print(f"连通分量大小分布 (前{min(max_components, n_components)}个):")

        total_shown = 0
        for comp_id, size in sorted_components[:max_components]:
            print(f"  分量 {comp_id}: {size} 个节点")
            total_shown += size

        if n_components > max_components:
            remaining_nodes = self.n_nodes - total_shown
            print(
                f"  其他 {n_components - max_components} 个分量: {remaining_nodes} 个节点"
            )

        # 检查是否连通
        is_connected = n_components == 1
        print(f"图是否连通: {is_connected}")

        elapsed_time = time.time() - start_time
        print(f"计算时间: {elapsed_time:.2f}秒")

        return n_components, labels, is_connected

    def centrality_analysis(self, top_k=10):
        """优化的中心性分析"""
        print("\n" + "=" * 50)
        print("中心性分析 (优化版)")
        print("=" * 50)

        start_time = time.time()

        # 只计算度中心性（其他中心性计算成本太高）
        degrees = np.array(self.adj_matrix.sum(axis=1)).flatten()
        degree_centrality = degrees / (self.n_nodes - 1)

        print(f"度中心性最高的 {top_k} 个节点:")
        top_indices = np.argpartition(degree_centrality, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(degree_centrality[top_indices])[::-1]]

        for i, node in enumerate(top_indices):
            print(f"  {i+1:2d}. 节点 {node:5d}: {degree_centrality[node]:.6f}")

        elapsed_time = time.time() - start_time
        print(f"计算时间: {elapsed_time:.2f}秒")

        return degree_centrality

    def sampling_based_clustering(self, sample_size=1000):
        """基于抽样的聚类系数分析"""
        print("\n" + "=" * 50)
        print("聚类系数分析 (抽样版)")
        print("=" * 50)

        start_time = time.time()

        # 抽样计算聚类系数
        if sample_size > self.n_nodes:
            sample_size = self.n_nodes

        sampled_indices = np.random.choice(self.n_nodes, sample_size, replace=False)
        clustering_coeffs = []

        for i in sampled_indices:
            neighbors = self.adj_matrix[i].nonzero()[1]
            k = len(neighbors)

            if k < 2:
                clustering_coeffs.append(0.0)
            else:
                # 计算邻居之间的边数
                edges_between_neighbors = 0
                # 只检查部分邻居对以避免组合爆炸
                max_pairs = min(1000, k * (k - 1) // 2)
                if k > 50:  # 对于高度数节点，进一步抽样
                    neighbor_pairs = []
                    for _ in range(max_pairs):
                        u, v = np.random.choice(neighbors, 2, replace=False)
                        if u != v and self.adj_matrix[u, v] > 0:
                            edges_between_neighbors += 1
                    coeff = (2 * edges_between_neighbors) / max_pairs
                else:
                    for u_idx, u in enumerate(neighbors):
                        for v in neighbors[u_idx + 1 :]:
                            if self.adj_matrix[u, v] > 0:
                                edges_between_neighbors += 1
                    coeff = (2 * edges_between_neighbors) / (k * (k - 1))

                clustering_coeffs.append(coeff)

        clustering_coeffs = np.array(clustering_coeffs)

        print(f"基于 {sample_size} 个节点的抽样结果:")
        print(f"平均聚类系数: {np.mean(clustering_coeffs):.6f}")
        print(f"聚类系数标准差: {np.std(clustering_coeffs):.6f}")
        print(f"聚类系数中位数: {np.median(clustering_coeffs):.6f}")

        elapsed_time = time.time() - start_time
        print(f"计算时间: {elapsed_time:.2f}秒")

        return clustering_coeffs

    def efficient_visualization(self, max_nodes=1000):
        """针对大图的简化可视化"""
        print("\n" + "=" * 50)
        print("简化可视化")
        print("=" * 50)

        # 如果图太大，只可视化最大连通分量或抽样
        n_components, labels, _ = self.connected_components_analysis(max_components=5)

        if n_components == 1 and self.n_nodes > max_nodes:
            print("图太大，进行抽样可视化...")
            # 随机抽样节点
            sample_nodes = np.random.choice(self.n_nodes, max_nodes, replace=False)
            subgraph = self.adj_matrix[sample_nodes, :][:, sample_nodes]
            G = nx.from_scipy_sparse_array(subgraph)
        else:
            # 使用最大连通分量
            component_sizes = Counter(labels)
            largest_component_id = max(component_sizes, key=component_sizes.get)
            nodes_in_largest = np.where(labels == largest_component_id)[0]

            if len(nodes_in_largest) > max_nodes:
                print(f"最大连通分量有 {len(nodes_in_largest)} 个节点，进行抽样...")
                nodes_in_largest = np.random.choice(
                    nodes_in_largest, max_nodes, replace=False
                )

            subgraph = self.adj_matrix[nodes_in_largest, :][:, nodes_in_largest]
            G = nx.from_scipy_sparse_array(subgraph)

        plt.figure(figsize=(15, 5))

        # 度分布直方图
        plt.subplot(1, 3, 1)
        degrees = [d for _, d in G.degree()]
        plt.hist(degrees, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        plt.xlabel("度")
        plt.ylabel("频率")
        plt.title("度分布")

        # 图结构可视化
        plt.subplot(1, 3, 2)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=20, alpha=0.6, edge_color="gray", width=0.5)
        plt.title("图结构")

        # 邻接矩阵热图（只显示部分）
        plt.subplot(1, 3, 3)
        if subgraph.shape[0] > 500:
            # 如果还是太大，进一步抽样
            sample_idx = np.random.choice(subgraph.shape[0], 500, replace=False)
            subgraph = subgraph[sample_idx, :][:, sample_idx]

        sns.heatmap(subgraph.toarray(), cmap="Blues", cbar=True)
        plt.title("邻接矩阵抽样")

        plt.tight_layout()
        plt.show()

    def memory_usage_report(self):
        """内存使用报告"""
        print("\n" + "=" * 50)
        print("内存使用报告")
        print("=" * 50)

        if sparse.issparse(self.adj_matrix):
            dense_size = self.n_nodes * self.n_nodes * 8 / (1024**3)  # GB
            sparse_size = (
                self.adj_matrix.data.nbytes
                + self.adj_matrix.indices.nbytes
                + self.adj_matrix.indptr.nbytes
            ) / (
                1024**3
            )  # GB

            print(f"稠密矩阵估计大小: {dense_size:.2f} GB")
            print(f"稀疏矩阵实际大小: {sparse_size:.2f} GB")
            print(f"内存节省: {(1 - sparse_size/dense_size)*100:.1f}%")

    def comprehensive_analysis(self, visualize=True, sample_size=1000):
        """执行全面的优化分析"""
        print("开始稀疏图探索性分析 (优化版)...")

        # 内存报告
        self.memory_usage_report()

        # 基本统计
        basic_stats = self.basic_statistics()

        # 度分析
        degrees, degree_stats = self.degree_analysis(sample_size)

        # 连通性分析
        connectivity_results = self.connected_components_analysis()

        # 中心性分析
        centrality = self.centrality_analysis()

        # 聚类系数分析（抽样）
        clustering = self.sampling_based_clustering(sample_size)

        # 简化可视化
        if visualize and self.n_nodes <= 10000:  # 只在节点数适中时可视化
            self.efficient_visualization()
        elif visualize:
            print("\n图太大，跳过详细可视化")
            if input("是否显示简化抽样可视化? (y/n): ").lower() == "y":
                self.efficient_visualization()

        # 返回所有分析结果
        return {
            "basic_stats": basic_stats,
            "degrees": degrees,
            "degree_stats": degree_stats,
            "connectivity": connectivity_results,
            "centrality": centrality,
            "clustering": clustering,
        }


def compute_communitude_metric(A, labels, axis=0):
    """
    Calculate the communitude metric for each community to compare intra-layer and inter-layer community quality.
    """
    A = np.array(A)
    labels = np.array(labels)
    total_edge_weight = np.sum(A)
    unique_communities = np.unique(labels)
    results = {}

    for ck in unique_communities:
        if axis == 0:
            rows_in_ck = np.where(labels == ck)[0]
            submatrix = A[np.ix_(rows_in_ck, list(range(A.shape[1])))]
            e_intra_ck = np.sum(submatrix)
            e_inter_ck = np.sum(A[rows_in_ck, :]) - e_intra_ck
        else:
            cols_in_ck = np.where(labels == ck)[0]
            submatrix = A[np.ix_(list(range(A.shape[0])), cols_in_ck)]
            e_intra_ck = np.sum(submatrix)
            e_inter_ck = np.sum(A[:, cols_in_ck]) - e_intra_ck

        if total_edge_weight == 0:
            results[ck] = 0.0
            continue

        numerator = (e_intra_ck / total_edge_weight) - (
            (e_intra_ck + e_inter_ck) / (2 * total_edge_weight)
        ) ** 2
        denominator = ((e_intra_ck + e_inter_ck) / (2 * total_edge_weight)) ** 2 * (
            1 - ((e_intra_ck + e_inter_ck) / (2 * total_edge_weight)) ** 2
        )

        results[ck] = 0.0 if denominator == 0 else numerator / denominator

    return results


def create_mapping(row):
    if row["type"] == "intra":
        return 100 + row["community_id"]
    else:
        return 200 + row["community_id"]


if __name__ == "__main__":
    from sklearn.decomposition import NMF

    np.random.seed(42)
    simulated = np.random.randint(0, 2, (10, 10))
    model = NMF(n_components=4, init="random", random_state=42)
    U = model.fit_transform(simulated)
    Vt = model.components_
    V = Vt.T
    print(U.shape)  # (10, 4)
    labels = np.argmax(U, axis=1)  # (10,)
    unique_labels = np.unique(labels)
    total_edge_weight = np.sum(simulated)
    # print(unique_labels)

    results = {}
    for ul in unique_labels:
        rows_in_ul = np.where(labels == ul)[0]
        # row_in_ul 是 ul 类别的所有行索引(对应节点id)
        # if ul == unique_labels[0]:
        #     print(np.where(labels == ul))
        # print(ul,rows_in_ul)
        submatrix = simulated[np.ix_(rows_in_ul, list(range(simulated.shape[1])))]
        # np.ix_ : 用于生成一个二维的索引数组，用于从矩阵中提取子矩阵
        # 这里的 np.ix_(rows_in_ul, list(range(simulated.shape[1]))) 表示提取 simulated 矩阵中 rows_in_ul 行和所有列的子矩阵
        # 即提取节点15连接到的点的索引(列索引)
        # 再用 np.where(submatrix == 1)[1] 找到节点15连接到的点的索引(列索引)
        # 即节点15连接到的点的索引(列索引) = [15, 22, 24, 31, 32, 34, 37, 40, 42, 44]
        # 再用 np.where(simulated[rows_in_ul, :] == 1)[1] 找到节点15连接到的点的索引(列索引)
        # 即节点15连接到的点的索引(列索引) = [15, 22, 24, 31, 32, 34, 37, 40, 42, 44]
        # 再用 np.where(U[rows_in_ul, :] == 1)[1] 找到节点15连接到的点的索引(列索引)
        # 即节点15连接到的点的索引(列索引) = [2, 3, 4, 9]
        # 这里的 np.ix_(rows_in_ul, list(range(U.shape[1]))) 表示提取 U 矩阵中 rows_in_ul 行和所有列的子矩阵
        # 即提取节点15连接到的点的索引(列索引) = [2, 3, 4, 9]
        e_intra_ul = np.sum(submatrix)
        e_inter_ul = np.sum(simulated[rows_in_ul, :]) - e_intra_ul
        if ul == unique_labels[0]:
            print(rows_in_ul)  # = [2 3 4 9]
            print(list(range(simulated.shape[1])))  # = [0, 1, 2, ..., 9]
            print(np.ix_(rows_in_ul, list(range(simulated.shape[1]))))
            # (array([[2],
            #        [3],
            #        [4],
            #        [9]]), array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]))

            print(submatrix)  # 索引 rows_in_ul 各行的元素，即节点连接到的点的索引
            # [[1 0 1 1 1 1 1 1 1 1]
            #  [0 0 1 1 1 0 1 0 0 0]
            #  [0 0 1 1 1 1 1 0 1 1]
            #  [1 1 1 1 1 1 1 1 1 0]]
            print(np.where(submatrix == 1))  # 找到节点15连接到的点的索引
            print(np.where(submatrix == 1)[1])  # 找到节点15连接到的点的索引(列索引)
