import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

from src.processor import feature_process,edge_process
from src.evaluator import Evaluator
def spectral_clustering(X, k):
    if  isinstance(X, csr_matrix):
        X = X.toarray()
    similarity_matrix = X
    # 计算度矩阵 D
    degree_matrix = np.sum(similarity_matrix, axis=1)
    D = np.diag(degree_matrix)
    
    # 计算规范化拉普拉斯矩阵 L_sym
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix))
    L = D - similarity_matrix
    L_norm = np.dot(np.dot(D_inv_sqrt, L), D_inv_sqrt)
    
    # 特征值分解
    eigvals, eigvecs = eigh(L_norm)
    
    # 选择最小的 k 个特征值对应的特征向量
    embedding = eigvecs[:, :k]
    
    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embedding)
    
    return labels

def nmf_clustering(X, k):
    if isinstance(X, csr_matrix):
        X = X.toarray()
    # 初始化 NMF 模型
    nmf = NMF(n_components=k, init='random', random_state=42)
    
    # 对数据进行非负矩阵分解
    W = nmf.fit_transform(X)
    
    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(W)
    
    return labels

if __name__ == "__main__":
    X = feature_process("cora")
    Y = edge_process("cora")
    k = 7
    targets = pd.read_csv(f"data/graphs/cora.targets", header=None).values[:, 1]
    print("Spectral Clustering Metrics:")
    print("feature:")
    spectral_labels = spectral_clustering(X, k)
    evaluator = Evaluator(targets, spectral_labels)    
    evaluator.print_metrics()
    print("edge:")
    spectral_labels = spectral_clustering(Y, k)
    evaluator = Evaluator(targets, spectral_labels)    
    evaluator.print_metrics()
    print("NMF Clustering Metrics:")
    print("feature:")
    nmf_labels = nmf_clustering(X, k)
    evaluator = Evaluator(targets, nmf_labels)
    evaluator.print_metrics()
    print("edge:")
    nmf_labels = nmf_clustering(Y, k)
    evaluator = Evaluator(targets, nmf_labels)
    evaluator.print_metrics()

