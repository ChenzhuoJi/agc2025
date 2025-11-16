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
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
import networkx as nx
from sklearn.decomposition import NMF
from collections import Counter, defaultdict

import warnings

warnings.filterwarnings("ignore")

# 设置matplotlib中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 指定默认字体
plt.rcParams["axes.unicode_minus"] = False  # 解决保存图像时负号'-'显示为方块的问题


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


def json2long(json_input_path, long_output_path):
    """
    将 JSON {"节点ID": [拥有的属性ID列表]} 转换为长格式。
    输出格式：每行 "节点ID\t属性ID"。
    排序方式：首先按属性ID升序，然后按节点ID升序。

    :param json_input_path: 输入的 JSON 文件路径。
    :param long_output_path: 输出的长格式文件路径。
    """
    print(f"[*] 开始转换: {json_input_path} -> {long_output_path}")

    # 1. 读取 JSON 文件
    try:
        with open(json_input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[!] 错误: 输入文件 '{json_input_path}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"[!] 错误: 输入文件 '{json_input_path}' 不是有效的 JSON 格式。")
        return

    # 2. 数据预处理和收集
    # 创建一个字典，key为属性ID，value为拥有该属性的节点ID列表
    attr_to_nodes = defaultdict(list)

    all_node_ids = set()
    all_attr_ids = set()

    for node_id_str, attr_id_list in data.items():
        node_id_int = int(node_id_str)
        all_node_ids.add(node_id_int)

        for attr_id in attr_id_list:
            attr_id_int = int(attr_id)
            attr_to_nodes[attr_id_int].append(node_id_int)
            all_attr_ids.add(attr_id_int)

    # 3. 排序
    # 对所有属性ID进行升序排序
    sorted_attr_ids = sorted(list(all_attr_ids))

    # 对每个属性对应的节点ID列表进行升序排序
    for attr_id in sorted_attr_ids:
        attr_to_nodes[attr_id].sort()

    print(f"[*] 发现 {len(all_node_ids)} 个节点, {len(all_attr_ids)} 个独特属性。")

    # 4. 写入长格式文件
    total_lines = 0
    try:
        with open(long_output_path, "w", encoding="utf-8") as f:
            # 遍历排序后的属性ID
            for attr_id in sorted_attr_ids:
                # 遍历当前属性下排序后的节点ID列表
                for node_id in attr_to_nodes[attr_id]:
                    f.write(f"{node_id}\t{attr_id}\n")
                    total_lines += 1
        print(
            f"[✓] 转换成功! 共生成 {total_lines} 条记录，已保存到 '{long_output_path}'"
        )

    except IOError as e:
        print(f"[!] 错误: 写入文件 '{long_output_path}' 失败。 {e}")


def compute_AS_with_NMF(A, r, random_state=42):
    """
    使用非负矩阵分解(NMF)计算给定邻接矩阵(或相似度矩阵)A的非对称惊喜度(Asymmetric Surprise, AS)。

    参数:
        A (numpy.ndarray): 图的邻接矩阵（或相似度矩阵），形状为(n×n)，其中n是节点数
        r (int): 设定的要检测的社区数量
        random_state (int, optional): 随机数生成器的种子，用于确保结果的可重复性，默认为42

    返回:
        tuple: 包含两个元素的元组
            - float: 计算得到的非对称惊喜度(AS)值
            - numpy.ndarray: 每个节点所属的社区标签，形状为(n,)

    算法原理:
        1. 使用NMF将邻接矩阵分解为两个非负矩阵的乘积，从而获得节点的社区隶属度
        2. 基于分解结果确定每个节点的社区标签
        3. 计算实际的社区内边比例与随机分布下的期望社区内边比例
        4. 使用KL散度计算这两个分布之间的差异，得到非对称惊喜度
    """
    # 获取图中节点的数量
    n = A.shape[0]

    # -------- Step 1: NMF Decomposition --------
    # 初始化NMF模型，使用NNDSVD(非负双重奇异值分解)作为初始化方法
    # 设置最大迭代次数为500以确保收敛
    nmf = NMF(n_components=r, init="nndsvd", random_state=random_state, max_iter=500)

    # 对邻接矩阵A进行NMF分解，得到节点-社区隶属度矩阵U(n×k)
    U = nmf.fit_transform(A)  # n x k

    # 对每个节点，选择隶属度最大的社区作为其标签
    labels = np.argmax(U, axis=1)  # 取最大隶属度的社区作为标签

    # -------- Step 2: compute community structure --------
    # 计算图中边的总数（因为邻接矩阵是对称的，所以需要除以2）
    E = np.sum(A) / 2  # total number of edges

    # 计算社区内部实际存在的边数
    E_intra = 0
    for c in set(labels):
        # 获取属于当前社区c的所有节点的索引
        idx = np.where(labels == c)[0]
        # 提取这些节点组成的子图的邻接矩阵
        subgraph = A[np.ix_(idx, idx)]
        # 累加子图中的边数（同样除以2避免重复计算）
        E_intra += np.sum(subgraph) / 2

    # 计算实际的社区内边比例q
    q = E_intra / E if E > 0 else 0

    # 计算随机分布下期望的社区内边比例q_exp
    # 首先计算每个社区的大小
    sizes = [np.sum(labels == c) for c in set(labels)]
    # 计算期望的社区内边比例：所有社区可能的内部边数之和除以图中可能的总边数
    q_exp = sum(s * (s - 1) / 2 for s in sizes) / (n * (n - 1) / 2)

    # 处理边界情况：如果q或q_exp为0或1，则KL散度无法定义，返回0
    if q in [0, 1] or q_exp in [0, 1]:
        return 0.0, labels

    # 计算KL散度：衡量实际分布与期望分布之间的差异
    KL = q * np.log(q / q_exp) + (1 - q) * np.log((1 - q) / (1 - q_exp))

    # 计算非对称惊喜度：KL散度乘以2倍的边数
    AS = 2 * E * KL

    # 返回计算得到的非对称惊喜度和节点社区标签
    return AS, labels


def standardize_feature_ids(graphs_dir, output_dir="st"):
    """
    检测特征ID不连续或不从0开始的文件，并将其标准化（从0开始，连续化）

    Args:
        graphs_dir (str): graphs目录路径
        output_dir (str): 输出目录名称，默认为'st'

    Returns:
        dict: 处理结果
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = {}

    # 遍历graphs目录下的所有.features文件
    for filename in os.listdir(graphs_dir):
        if filename.endswith(".features"):
            file_path = os.path.join(graphs_dir, filename)

            try:
                # 读取.features文件
                with open(file_path, "r") as f:
                    features_data = json.load(f)

                # 收集所有特征ID
                all_feature_ids = set()
                for node_features in features_data.values():
                    all_feature_ids.update(node_features)

                # 检查特征ID是否连续且从0开始
                all_feature_ids = sorted(list(all_feature_ids))
                expected_ids = list(range(len(all_feature_ids)))

                # 如果特征ID已经是从0开始且连续的，则直接复制文件
                if all_feature_ids == expected_ids:
                    # 直接复制文件到st目录
                    output_file_path = os.path.join(output_dir, filename)
                    shutil.copy(file_path, output_file_path)

                    # 创建节点ID映射文件（保持不变）
                    node_mapping = {
                        node_id: node_id for node_id in features_data.keys()
                    }
                    mapping_file_path = os.path.join(
                        output_dir, filename.replace(".features", ".node_mapping.json")
                    )
                    with open(mapping_file_path, "w") as f:
                        json.dump(node_mapping, f, indent=2)

                    results[filename] = {
                        "status": "already_standardized",
                        "original_feature_count": len(all_feature_ids),
                        "mapping_file": mapping_file_path,
                    }
                else:
                    # 需要标准化特征ID
                    # 创建特征ID映射（原ID -> 新ID）
                    feature_id_mapping = {
                        old_id: new_id for new_id, old_id in enumerate(all_feature_ids)
                    }

                    # 创建标准化后的数据
                    standardized_data = {}
                    node_mapping = {}  # 节点ID映射（如果需要的话）

                    for node_idx, (node_id, node_features) in enumerate(
                        features_data.items()
                    ):
                        # 保持节点ID不变，只标准化特征ID
                        standardized_features = [
                            feature_id_mapping[feat_id] for feat_id in node_features
                        ]
                        standardized_data[node_id] = standardized_features
                        node_mapping[node_id] = node_idx  # 如果需要重新编号节点

                    # 保存标准化后的.features文件
                    output_file_path = os.path.join(output_dir, filename)
                    with open(output_file_path, "w") as f:
                        json.dump(standardized_data, f, indent=2)

                    # 保存特征ID映射关系
                    mapping_info = {
                        "feature_id_mapping": feature_id_mapping,
                        "node_mapping": node_mapping,
                        "original_min_feature_id": (
                            min(all_feature_ids) if all_feature_ids else None
                        ),
                        "original_max_feature_id": (
                            max(all_feature_ids) if all_feature_ids else None
                        ),
                        "new_feature_count": len(all_feature_ids),
                    }

                    mapping_file_path = os.path.join(
                        output_dir, filename.replace(".features", ".mapping.json")
                    )
                    with open(mapping_file_path, "w") as f:
                        json.dump(mapping_info, f, indent=2)

                    results[filename] = {
                        "status": "standardized",
                        "original_feature_count": len(all_feature_ids),
                        "new_feature_count": len(all_feature_ids),
                        "min_feature_id": (
                            min(all_feature_ids) if all_feature_ids else None
                        ),
                        "max_feature_id": (
                            max(all_feature_ids) if all_feature_ids else None
                        ),
                        "output_file": output_file_path,
                        "mapping_file": mapping_file_path,
                    }

            except Exception as e:
                results[filename] = {"status": "error", "error": str(e)}

    return results


def determine_community_number(A, max_r=10):
    """
    使用非对称惊喜度(AS)指标自动确定图的最优社区数量。

    参数:
        A (numpy.ndarray): 图的邻接矩阵，形状为(n×n)，其中n是节点数
        max_r (int, optional): 尝试的最大社区数量，默认值为10

    返回:
        int: 使AS值最大的最优社区数量

    算法原理:
        1. 遍历从2到max_r的所有可能社区数量k
        2. 对每个k值，使用compute_AS_with_NMF计算对应的非对称惊喜度(AS)值
        3. 找出AS值最大的k值，作为最优社区数量

    注意事项:        - 社区数量通常从2开始尝试，因为单个社区没有划分意义
        - 当存在多个k值对应相同的最大AS值时，返回第一个出现的k值
    """
    # 初始化列表用于存储不同社区数量对应的AS值
    AS_values = []

    # 遍历从2到max_r的所有可能社区数量k
    # 2是社区数量的最小有意义值，单个社区无法体现社区划分
    for k in range(2, max_r + 1):
        # 计算当前社区数量k对应的非对称惊喜度值
        # 使用固定的random_state=42确保结果的可重复性
        AS_values.append(compute_AS_with_NMF(A, k, random_state=42))

    # 找出所有AS值中的最大值
    max_AS = max(AS_values)

    # 遍历AS值列表，找出第一个等于最大值的索引
    for index, value in enumerate(AS_values):
        if value == max_AS:
            # 将索引转换为对应的社区数量k（索引从0开始，对应k=2）
            r = index + 2
            # 返回最优社区数量
            return r


def check_featjson(featdict):
    # 验证节点id是否从零开始且连续
    node_ids = sorted(list(int(node_id) for node_id in featdict.keys()))
    if not all(int(node_id) == i for i, node_id in enumerate(node_ids)):
        max_id = max(int(node_id) for node_id in node_ids)
        min_id = min(int(node_id) for node_id in node_ids)
        print(f"最大节点ID: {max_id}")
        print(f"最小节点ID: {min_id}")
        print(f"预期节点数: {max_id - min_id + 1}, 实际节点数: {len(node_ids)}")
    # 验证属性id是否从零开始且连续
    all_feat = set()
    for node, features in featdict.items():
        all_feat.update(features)
    all_feat = sorted(list(all_feat))
    if not all(int(feat_id) == i for i, feat_id in enumerate(all_feat)):
        print(f"最大属性ID: {max(int(feat_id) for feat_id in all_feat)}")
        print(f"最小属性ID: {min(int(feat_id) for feat_id in all_feat)}")
        print(
            f"预期属性数: {max(int(feat_id) for feat_id in all_feat) - min(int(feat_id) for feat_id in all_feat) + 1}, 实际属性数: {len(all_feat)}"
        )
        raise ValueError("Feature IDs must be consecutive integers starting from 0")
    pass


def check_edges(edgesframe):
    # 验证是否从零开始且连续
    edges = np.unique(np.array(edgesframe, dtype=int))
    max_id = max(edges.max(), edges.min())
    min_id = min(edges.max(), edges.min())
    if not all(
        int(node_id) == i for i, node_id in enumerate(range(min_id, max_id + 1))
    ):
        raise ValueError("Edge IDs must be consecutive integers starting from 0")


def json2featmat(file_path=None):
    with open(file_path, "r") as f:
        # 解析JSON文件
        features_dict = json.load(f)

    # 准备数据结构
    row_indices = []  # 行索引（节点ID）
    col_indices = []  # 列索引（特征ID）
    data = []  # 数据值（这里都是1）

    # 获取所有节点和特征
    nodes = list(features_dict.keys())

    # 确定最大特征ID
    max_feature_id = 0
    for node, features in features_dict.items():
        if features:
            current_max = max(features)
            if current_max > max_feature_id:
                max_feature_id = current_max

    # 构建稀疏矩阵数据
    for node_idx, node in enumerate(nodes):
        for feature_id in features_dict[node]:
            row_indices.append(node_idx)
            col_indices.append(feature_id)
            data.append(1.0)  # 存在特征值为1

    # 创建稀疏矩阵（COO格式）
    num_nodes = len(nodes)
    num_features = max_feature_id + 1

    coo = sparse.coo_matrix(
        (data, (row_indices, col_indices)), shape=(num_nodes, num_features)
    )
    csr = coo.tocsr()
    return csr


if __name__ == "__main__":
    # standardize_feature_ids('stgraphs')

    for file in os.listdir("stgraphs"):
        if file.endswith(".features"):
            dataname = file.split(".")[0]
            with open(f"stgraphs/{file}", "r") as f:
                print(f"checking {dataname}")
                features_dict = json.load(f)
            edgesframe = pd.read_csv(f"stgraphs/{dataname}.edges", header=None)
            check_edges(edgesframe)
            check_featjson(features_dict)
            # json2featmat(f"stgraphs/{file}")
