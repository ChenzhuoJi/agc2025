import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from src.ml_jnmf import ML_JNMF
from main import precompute, load_data


# ----------------------------------------------------
#  构造一致性矩阵函数
# ----------------------------------------------------
def consensus_matrix(labels_list):
    """
    构建聚类一致性矩阵 (Consensus Matrix)

    该函数通过分析多个聚类结果，计算样本对在同一聚类中出现的频率，
    用于评估聚类结果的稳定性和一致性。

    参数:
    -----------
    labels_list : list of ndarray
        一个包含 B 个标签数组的列表，每个标签数组长度为 n
        每个数组表示一次独立的聚类结果

    返回:
    -----------
    M : ndarray, shape (n, n)
        一致性矩阵，元素值范围 [0, 1]
        M[i,j] 表示在所有聚类结果中，样本 i 和样本 j 被分配到同一簇的比例

    算法原理:
    -----------
    1. 对每个聚类结果，构建一个 n×n 的共聚类矩阵
       - 如果样本 i 和 j 在同一簇，则对应位置为 1，否则为 0
    2. 累加所有聚类结果的共聚类矩阵
    3. 除以聚类次数 B，得到一致性比例

    示例:
    -----------
    >>> labels_list = [
    ...     np.array([0, 0, 1, 1]),  # 第一次聚类结果
    ...     np.array([0, 0, 1, 1]),  # 第二次聚类结果
    ...     np.array([1, 1, 0, 0])   # 第三次聚类结果
    ... ]
    >>> consensus_matrix(labels_list)
    array([[1.        , 1.        , 0.        , 0.        ],
           [1.        , 1.        , 0.        , 0.        ],
           [0.        , 0.        , 1.        , 1.        ],
           [0.        , 0.        , 1.        , 1.        ]])
    """
    # 获取样本数量和聚类次数
    n = len(labels_list[0])  # 样本数量
    B = len(labels_list)  # 聚类次数

    # 初始化一致性矩阵
    M = np.zeros((n, n))

    # 遍历每个聚类结果
    for labels in labels_list:
        # 使用广播机制构建共聚类矩阵
        # labels[:, None] 变成列向量 (n, 1)
        # labels[None, :] 变成行向量 (1, n)
        # 相等比较生成 n×n 的布尔矩阵，表示样本对是否在同一簇
        M += (labels[:, None] == labels[None, :]).astype(int)

    # 计算一致性比例：除以聚类次数 B
    return M / B


# ----------------------------------------------------
#  计算 PAC 指标函数
# ----------------------------------------------------
def pac_score(M, u1=0.1, u2=0.9):
    """
    计算PAC (Proportion of Ambiguous Clustering) 指标

    PAC指标用于评估一致性矩阵的清晰程度，通过计算处于模糊区间的元素比例
    来衡量聚类结果的稳定性。PAC值越小，表示聚类结果越稳定、越清晰。

    参数:
    -----------
    M : ndarray, shape (n, n)
        一致性矩阵，元素值范围 [0, 1]
        通常由consensus_matrix函数生成
    u1 : float, default=0.1
        模糊区间的下界阈值
    u2 : float, default=0.9
        模糊区间的上界阈值

    返回:
    -----------
    ambiguous : float
        PAC指标值，范围 [0, 1]
        表示处于模糊区间 (u1, u2) 的元素比例

    算法原理:
    -----------
    1. 提取一致性矩阵的下三角元素（避免重复计算）
    2. 统计处于模糊区间 (u1, u2) 的元素比例
    3. PAC值越小，说明聚类结果越稳定

    数学解释:
    -----------
    - 一致性矩阵元素接近0：样本对很少被分到同一簇
    - 一致性矩阵元素接近1：样本对总是被分到同一簇
    - 一致性矩阵元素接近0.5：样本对有时被分到同一簇，存在歧义

    使用场景:
    -----------
    - 确定最优聚类数量：选择PAC值最小的聚类数
    - 评估聚类算法稳定性：PAC值越小，算法越稳定
    - 比较不同聚类方法：PAC值可用于选择最佳聚类算法

    示例:
    -----------
    >>> M = np.array([[1.0, 0.8, 0.1],
    ...               [0.8, 1.0, 0.9],
    ...               [0.1, 0.9, 1.0]])
    >>> pac_score(M, u1=0.1, u2=0.9)
    0.3333333333333333  # 表示33.3%的元素处于模糊区间

    注意事项:
    -----------
    - 只使用下三角元素避免重复计算（矩阵是对称的）
    - 默认阈值 u1=0.1, u2=0.9 可根据实际需求调整
    - 对于完全清晰的聚类，PAC值应接近0
    - 对于完全模糊的聚类，PAC值可能接近1
    """
    # 获取矩阵维度（样本数量）
    n = M.shape[0]

    # 提取下三角元素（不包括对角线）
    # np.tril_indices(n, k=-1) 生成下三角元素的索引
    # k=-1 表示不包括主对角线元素
    tril_vals = M[np.tril_indices(n, k=-1)]

    # 统计处于模糊区间的元素
    # np.logical_and 确保元素同时满足大于u1和小于u2
    # .mean() 计算满足条件的元素比例
    ambiguous = np.logical_and(tril_vals > u1, tril_vals < u2).mean()

    return ambiguous


def calculate_r(la, ls, li, dataname, max_r=10, B=3):
    consensus_examples = {}
    pac_values = []
    r_values = list(range(2, max_r + 1))
    for r in r_values:
        print(f"r={r}")
        labels_list = []
        for b in range(B):
            # 随机初始化种子在这行
            model = ML_JNMF(la, ls, li, random_state=b)
            cluster_labels = model.fit_predict(r, pred_method="lambda")
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
    plt.plot(r_values, pac_values, marker="o")
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
    # 如果没有预计算
    from main import precompute
    import os
    os.makedirs("precomputed", exist_ok=True)
    params = {'feature_kernel':'rbf','order':5,'decay':2}
    precompute('cora',params)
    # 示例数据
    # label_list=[np.array([7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 2, 2,
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

    la = np.load(r"precomputed\cora_la.npy")
    ls = np.load(r"precomputed\cora_ls.npy")
    li = np.load(r"precomputed\cora_li.npy")
    # 计算 r
    # 这次上传在ML_JNMF.matrixInit中用的是random随机初始化，在154行指定了随机化种子
    r = calculate_r(la, ls, li, "cora")
    print(r)
