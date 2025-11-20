import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from src.ml_jnmf import ML_JNMF
from main import precompute,load_data
# ----------------------------------------------------
# 1. 生成示例数据
# ----------------------------------------------------
# X, y_true = make_blobs(
#     n_samples=200,
#     centers=5,
#     n_features=2,
#     random_state=42
# )


# ----------------------------------------------------
# 2. 构造一致性矩阵函数
# ----------------------------------------------------
def consensus_matrix(labels_list):
    """
    labels_list: 一个包含 B 个标签数组的列表，每个标签长度为 n
    返回：一致性矩阵 M (n × n)
    """
    n = len(labels_list[0])
    B = len(labels_list)

    M = np.zeros((n, n))

    for labels in labels_list:
        # labels[:,None] 变成 (n,1)
        # labels[None,:] 变成 (1,n)
        # 相等取 1，否则取 0
        # 这里M[i,j]加的是 第b个聚类结果中，i和j样本是否被分配到同一个簇的布尔值
        M += (labels[:, None] == labels[None, :]).astype(int)

    return M / B


# ----------------------------------------------------
# 3. 计算 PAC 指标函数
# ----------------------------------------------------
def pac_score(M, u1=0.1, u2=0.9):
    """
    PAC = 处于 (u1, u2) 区间的元素比例（仅用下三角）
    """
    n = M.shape[0]
    tril_vals = M[np.tril_indices(n, k=-1)]
    ambiguous = np.logical_and(tril_vals > u1, tril_vals < u2).mean()
    return ambiguous


# # ----------------------------------------------------
# # 4. 多次运行 KMeans，得到每个 r 的 PAC 值
# # ----------------------------------------------------
# r_values = list(range(2, 8))   # 尝试 r=2~7
# B = 5                          # 每个 r 重复聚类次数

# pac_values = []
# consensus_examples = {}

# for r in r_values:
#     labels_list = []
#     for b in range(B):
#         km = KMeans(
#             n_clusters=r,
#             init="random",
#             n_init=1,
#             algorithm='lloyd',
#             random_state=b
#         )
#         labels = km.fit_predict(X)
#         labels_list.append(labels)

#     # 一致性矩阵
#     M = consensus_matrix(labels_list)
#     consensus_examples[r] = M

#     # PAC
#     pac_values.append(pac_score(M))
# la = np.load(r'precomputed\cora_la.npy')
# ls = np.load(r'precomputed\cora_ls.npy')
# li = np.load(r'precomputed\cora_li.npy')

def calculate_r(la,ls,li,dataname,max_r=10,B=3):
    consensus_examples = {}
    pac_values = []
    r_values = list(range(2, max_r+1))
    for r in r_values:
        print(f'r={r}')
        labels_list = []
        for b in range(B):
            model = ML_JNMF(la,ls,li,random_state=b)
            cluster_labels = model.fit_predict(r,pred_method='lambda')
            labels_list.append(cluster_labels)
        # 一致性矩阵
        M = consensus_matrix(labels_list)
        
        consensus_examples[r] = M

        # PAC
        pac_values.append(pac_score(M))
    # 找到最小 PAC 对应的 r, 注意索引要加 2
    r = pac_values.index(min(pac_values)) + 2
    print(pac_values)
    return r


def plot_pac_vs_r(r_values, pac_values, dataname):
    plt.figure(figsize=(6, 4))
    plt.plot(r_values, pac_values, marker='o')
    plt.xlabel("Number of clusters r")
    plt.ylabel("PAC score")
    plt.title(f"PAC vs r ({dataname})")
    plt.grid(True)
    plt.savefig(f"results/best_r/pac_vs_r_{dataname}.png")

    # plt.figure(figsize=(5, 4))
    # plt.imshow(consensus_examples[5])
    # plt.title("Consensus Matrix (r=5)")
    # plt.colorbar()
    # plt.show()
# # ----------------------------------------------------
# # 5. 绘制 PAC vs r 曲线
# # ----------------------------------------------------
# plt.figure(figsize=(6, 4))
# plt.plot(r_values, pac_values, marker='o')
# plt.xlabel("Number of clusters r")
# plt.ylabel("PAC score")
# plt.title("PAC vs r")
# plt.grid(True)
# plt.show()


# # ----------------------------------------------------
# # 6. 画一个一致性矩阵（以 r=5 为例）
# # ----------------------------------------------------
# plt.figure(figsize=(5, 4))
# plt.imshow(consensus_examples[5])
# plt.title("Consensus Matrix (r=5)")
# plt.colorbar()
# plt.show()
import networkx as nx
import numpy as np

def generate_graph(n, r, p_in=0.5, p_out=0.01, seed=42):
    """
    生成一个可控节点数量 n、社区数量 r 的无向图 (SBM)
    
    n: 总节点数
    r: 社区数
    p_in: 社区内部连边概率
    p_out: 社区间连边概率
    """
    np.random.seed(seed)
    
    # 每个社区的节点数量（尽量平均）
    sizes = [n // r] * r
    for i in range(n % r):  # 处理 n 不能整除 r 的情况
        sizes[i] += 1

    # 概率矩阵：社区内 p_in，社区间 p_out
    probs = [[p_in if i == j else p_out for j in range(r)] for i in range(r)]

    # 生成 SBM 图
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    A = nx.to_numpy_array(G)
    return A

if __name__ == "__main__":
    from main import precompute
    precompute('cora')
    # label_list=[np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2,
    #    2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1,
    #    1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2,
    #    2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1,
    #    1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]), np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2,
    #    2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 4, 4, 4, 4,
    #    4, 4, 4, 4, 4, 4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0,
    #    0, 0, 0, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1,
    #    1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])]
    # M = consensus_matrix(label_list)
    # print(M)
    # pac = pac_score(M)
    # print(pac)
    la = np.load(r'precomputed\cora_la.npy')
    ls = np.load(r'precomputed\cora_ls.npy')
    li = np.load(r'precomputed\cora_li.npy')
    r = calculate_r(la,ls,li,'cora')
    print(r)