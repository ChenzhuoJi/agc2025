import networkx as nx

# LFR基准图的参数设置
n = 1000  # 节点数量
tau1 = 2  # 节点度分布的幂律指数
tau2 = 1.5  # 社区大小分布的幂律指数（可以试着调低这个值）
mu = 0.1  # 混合参数（社区内外连接比例）
average_degree = 20  # 平均度数
min_community = 30  # 增加最小社区大小
max_community = 100  # 增加最大社区大小

# 生成LFR基准图
G = nx.generators.community.LFR_benchmark_graph(
    n=n, tau1=tau1, tau2=tau2, mu=mu, average_degree=average_degree,
    min_community=min_community, max_community=max_community, max_iters=1000, seed=42
)

# 打印图的信息
print(nx.info(G))

# 可视化部分（如果需要）
import matplotlib.pyplot as plt
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=False, node_size=20, alpha=0.6)
plt.show()
