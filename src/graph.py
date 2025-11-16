from typing import Literal
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from networkx.algorithms.community import greedy_modularity_communities
from src.processor import feature_process

import matplotlib.pyplot as plt


def visualize_connected_components(G, layout="spring", figsize=(10, 8)):
    # 获取所有连通分量
    connected_components = list(nx.connected_components(G))

    # 设置绘图大小
    plt.figure(figsize=figsize)

    # 遍历每个连通分量并绘制
    for i, component in enumerate(connected_components):
        # 创建一个子图，提取连通分量的子图
        subgraph = G.subgraph(component)

        # 设置布局
        if layout == "spring":
            pos = nx.spring_layout(subgraph)
        elif layout == "circular":
            pos = nx.circular_layout(subgraph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(subgraph)
        else:
            raise ValueError(f"Unsupported layout type: {layout}")

        # 绘制子图
        plt.subplot(5, 5, i + 1)  # 可以调整subplot的行列数
        nx.draw_networkx(
            subgraph,
            pos,
            node_size=50,
            with_labels=False,
            node_color="skyblue",
            edge_color="gray",
            alpha=0.7,
        )
        plt.title(f"Component {i + 1}")

    plt.tight_layout()
    plt.show()


def visualize_graph_with_degree(G, layout="spring", figsize=(10, 8)):
    # 获取每个节点的度数
    degrees = dict(G.degree())
    node_sizes = [degrees[node] * 10 for node in G.nodes()]  # 调整大小比例因子
    node_colors = list(degrees.values())  # 基于度数着色

    # 设置绘图大小
    plt.figure(figsize=figsize)

    # 设置布局
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unsupported layout type: {layout}")

    # 绘制图
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Blues,
        alpha=0.8,
    )
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # 显示图形
    plt.title("Graph with Node Degree Based Size and Color")
    plt.axis("off")
    plt.show()


def visualize_community_detection(
    G, layout: Literal["spring", "circular", "kamada_kawai"] = "spring", figsize=(10, 8)
):
    # 使用社区检测算法（贪婪模块度优化）
    communities = list(greedy_modularity_communities(G))

    # 设置绘图大小
    plt.figure(figsize=figsize)

    # 设置布局
    if layout == "spring":
        pos = nx.spring_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unsupported layout type: {layout}")

    # 为每个社区分配不同的颜色
    community_colors = plt.cm.get_cmap("Set1", len(communities))  # 使用多种颜色

    # 绘制图
    for i, community in enumerate(communities):
        # 获取社区中节点的集合
        community_nodes = list(community)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=community_nodes,
            node_size=10,
            node_color=[community_colors(i)],
            alpha=0.8,
        )

    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7)

    # 显示图形
    plt.title("Graph with Community Detection")
    plt.axis("off")
    plt.show()


def visualize_subgraph(G, center_node, radius=2, layout="spring", figsize=(8, 6)):
    # 获取一个给定节点的邻居（可以设置半径来确定范围）
    subgraph_nodes = nx.single_source_shortest_path_length(
        G, center_node, cutoff=radius
    )
    subgraph = G.subgraph(subgraph_nodes.keys())

    # 设置绘图大小
    plt.figure(figsize=figsize)

    # 设置布局
    if layout == "spring":
        pos = nx.spring_layout(subgraph)
    elif layout == "circular":
        pos = nx.circular_layout(subgraph)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subgraph)
    else:
        raise ValueError(f"Unsupported layout type: {layout}")

    # 绘制图
    nx.draw_networkx_nodes(
        subgraph, pos, node_size=100, node_color="lightblue", alpha=0.8
    )
    nx.draw_networkx_edges(subgraph, pos, edge_color="gray", alpha=0.7)
    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight="bold")

    # 显示图形
    plt.title(f"Subgraph of Node {center_node} and its {radius} Hop Neighbors")
    plt.axis("off")
    plt.show()


def graph_statistics(G):
    stats = {}

    # 基本信息
    stats["Number of nodes"] = G.number_of_nodes()
    stats["Number of edges"] = G.number_of_edges()

    # 节点度数统计
    degrees = [deg for node, deg in G.degree()]
    stats["Average degree"] = sum(degrees) / len(degrees) if degrees else 0
    stats["Max degree"] = max(degrees) if degrees else 0
    stats["Min degree"] = min(degrees) if degrees else 0

    # 图的密度 (Density)
    stats["Density"] = nx.density(G)

    # 连通性相关
    # stats['Connected components'] = list(nx.connected_components(G))
    stats["Number of connected components"] = len(list(nx.connected_components(G)))

    # 节点的度中心性（可选）
    # stats['Degree centrality'] = nx.degree_centrality(G)

    # 节点的介数中心性（可选）
    # stats['Betweenness centrality'] = nx.betweenness_centrality(G)

    # 图的直径和平均路径长度（仅对于连通图有效）
    if nx.is_connected(G):
        stats["Diameter"] = nx.diameter(G)
        stats["Average shortest path length"] = nx.average_shortest_path_length(G)
    else:
        stats["Diameter"] = "N/A (Graph is not connected)"
        stats["Average shortest path length"] = "N/A (Graph is not connected)"

    return stats


def advanced_graph_analysis(G):
    analysis_results = {}

    # 1. 图的谱分析（计算拉普拉斯矩阵的特征值）
    laplacian_matrix = nx.laplacian_matrix(G).todense()
    eigenvalues, _ = eigsh(laplacian_matrix, k=6, which="SM")  # 计算前6个最小的特征值
    analysis_results["Laplacian eigenvalues"] = eigenvalues.tolist()

    # 2. 社区检测（使用贪心模块度优化算法）
    communities = list(greedy_modularity_communities(G))
    analysis_results["Number of communities"] = len(communities)
    analysis_results["Communities"] = [list(community) for community in communities]

    # 3. 图的直径和平均路径长度（仅对于连通图有效）
    if nx.is_connected(G):
        analysis_results["Diameter"] = nx.diameter(G)
        analysis_results["Average shortest path length"] = (
            nx.average_shortest_path_length(G)
        )
    else:
        analysis_results["Diameter"] = "N/A (Graph is not connected)"
        analysis_results["Average shortest path length"] = (
            "N/A (Graph is not connected)"
        )

    # 4. 最短路径（Dijkstra算法示例）
    # 选择任意两个节点并计算它们之间的最短路径（例如：节点 0 和节点 1）
    try:
        analysis_results["Shortest path (0, 1)"] = nx.shortest_path(
            G, source=0, target=1, weight="weight"
        )
    except nx.NetworkXNoPath:
        analysis_results["Shortest path (0, 1)"] = "No path exists"

    # 5. 聚类系数（全图的平均值和每个节点的聚类系数）
    analysis_results["Clustering coefficient (average)"] = nx.average_clustering(G)
    analysis_results["Clustering coefficient (per node)"] = nx.clustering(G)

    # 6. 计算图的度数直方图
    degree_histogram = nx.degree_histogram(G)
    analysis_results["Degree histogram"] = degree_histogram

    for key, value in analysis_results.items():
        print(f"{key}: {value}")
    return analysis_results


def read_edge_list(dataname: str) -> nx.Graph:
    # 创建一个无向图
    G = nx.Graph()

    # 打开文件并读取每一行
    with open(f"data/graphs/{dataname}.edges", "r") as file:
        for line in file:
            # 分割每行的节点对
            node1, node2 = map(int, line.strip().split(","))
            # 添加边到图中
            G.add_edge(node1, node2)

    return G


def adjacency_to_graph(adj: np.ndarray) -> nx.Graph:
    # 如果邻接矩阵不是对称的，可以先强制转为无向
    adj = np.array(adj)

    # 使用 NetworkX 的内置函数构建图
    G = nx.from_numpy_array(adj, create_using=nx.Graph)

    # 默认权重属性名称是 'weight'
    return G


def read_features(dataname: str) -> np.ndarray:
    # 读取特征矩阵
    similarity_matrix = np.array(feature_process(dataname))
    G = adjacency_to_graph(similarity_matrix)
    return G


if __name__ == "__main__":
    # 使用示例
    dataname = "cora"  # 替换为你的文件名
    print(f"Graph {dataname} structure layer:")
    G = read_edge_list(dataname)

    # # 打印图的统计信息
    # stats = graph_statistics(G)
    # for key, value in stats.items():
    #     print(f"{key}: {value}")
    visualize_community_detection(G, layout="kamada_kawai")
    # advanced_graph_analysis(G)
    # print(f'Graph {dataname} attribute layer: ')
    # G = read_features(dataname)
    # stats = graph_statistics(G)
    # for key, value in stats.items():
    #     print(f"{key}: {value}")
