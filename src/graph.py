import networkx as nx
import numpy as np
def read_edge_list(dataname: str)->nx.Graph:
    # 创建一个无向图
    G = nx.Graph()

    # 打开文件并读取每一行
    with open(f"data/graphs/{dataname}.edges", 'r') as file:
        for line in file:
            # 分割每行的节点对
            node1, node2 = map(int, line.strip().split(','))
            # 添加边到图中
            G.add_edge(node1, node2)

    return G

def adjacency_to_graph(adj: np.ndarray)->nx.Graph:
    # 如果邻接矩阵不是对称的，可以先强制转为无向
    adj = np.array(adj)
    
    # 使用 NetworkX 的内置函数构建图
    G = nx.from_numpy_array(adj, create_using=nx.Graph)
    
    # 默认权重属性名称是 'weight'
    return G

# 使用示例
dataname = 'cora'  # 替换为你的文件名
G = read_edge_list(dataname)

# 打印图的基本信息
print(f"Graph {dataname} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

