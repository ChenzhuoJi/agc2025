import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
def graph_statistics(adj_matrix, plot=True):
    """
    Display statistical features of a graph given its adjacency matrix.

    Args:
        adj_matrix (numpy.ndarray): Adjacency matrix (n x n)
        plot (bool): Whether to show degree distribution plot
    """
    # 转成 NetworkX 图
    G = nx.from_numpy_array(adj_matrix)
    n = G.number_of_nodes()
    m = G.number_of_edges()

    # 度相关信息
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)

    # 稀疏度
    density = nx.density(G)

    # 聚类系数与连通性
    clustering = nx.average_clustering(G)
    is_connected = nx.is_connected(G)
    avg_path_len = nx.average_shortest_path_length(G) if is_connected else None

    print("=== Graph Statistics ===")
    print(f"Nodes: {n}")
    print(f"Edges: {m}")
    print(f"Density: {density:.4f}")
    print(f"Average Degree: {avg_degree:.2f}")
    print(f"Max Degree: {max_degree}, Min Degree: {min_degree}")
    print(f"Average Clustering Coefficient: {clustering:.4f}")
    print(f"Connected: {is_connected}")
    if avg_path_len:
        print(f"Average Shortest Path Length: {avg_path_len:.4f}")
    print("=========================")

    if plot:
        plt.figure(figsize=(5, 3))
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), edgecolor="black")
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()