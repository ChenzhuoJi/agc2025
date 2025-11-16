"""
Author: Pumpkin🎃
Date:2025-10-31
Description:evaluator, 评估器
"""

import json
from itertools import combinations
from collections import defaultdict

import numpy as np
from scipy.special import comb
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure,
)

from scipy.optimize import linear_sum_assignment

from src.helpers import json2featmat


class internalEvaluator:
    def __init__(self, cluster_labels, edges: np.ndarray, features: np.ndarray):
        """
        聚类评估类：用于评估聚类结果的内部指标

        参数:
        cluster_labels (list or array): 每个节点的预测聚类标签
        edges (np.ndarray): 图的边数据，形状为 (num_edges, 2)
        features (np.ndarray): 节点特征矩阵，形状为 (num_nodes, feature_dim)
        """
        self.cluster_labels = np.array(cluster_labels)
        self.cluster_labels = np.squeeze(self.cluster_labels)
        if self.cluster_labels.ndim != 1:
            raise ValueError(
                f"cluster_labels 必须是一维数组，当前形状: {self.cluster_labels.shape}"
            )

        self.edges = np.array(edges)
        self.features = np.array(features)
        self.feature_dim = self.features.shape[1]
        self.num_nodes = len(self.cluster_labels)

        # 2. 预计算每个簇包含的节点集合
        self.pred_clusters = defaultdict(set)
        for node_id, label in enumerate(self.cluster_labels):
            if isinstance(label, (np.ndarray, list)):
                raise ValueError(
                    f"cluster_labels 元素不能是数组/列表，当前元素: {label}"
                )
            self.pred_clusters[label].add(node_id)
        self.cluster_ids = list(self.pred_clusters.keys())

        # 3. 预计算每个节点的邻居集合，加速边的查找
        self.neighbors = defaultdict(set)
        for u, v in self.edges:
            # 确保 u 和 v 是整数类型
            self.neighbors[int(u)].add(int(v))
            self.neighbors[int(v)].add(int(u))

    def normalized_homogeneity(self):
        """
        计算归一化同质性 (Normalized Homogeneity, NorHo)。

        公式: NorHo = (1 / (N * p)) * sum_{k} [n_k * H(C_k)]
        其中:
        - N 是总节点数。
        - p 是特征维度。
        - n_k 是第 k 个簇的节点数。
        - H(C_k) 是第 k 个簇的同质性，H(C_k) = -sum_{j=1 to p} c_{kj} * (1 - c_{kj})
        - c_{kj} 是第 k 个簇中，特征 j 为非零值的节点比例。

        返回:
        float: 归一化同质性的值。如果没有提供特征数据，则抛出异常。
        """
        if self.features is None or self.feature_dim == 0:
            raise RuntimeError("计算同质性需要特征数据，请在初始化时提供 features。")

        total_sum = 0.0
        N = self.num_nodes
        p = self.feature_dim

        for label in self.cluster_ids:
            cluster_nodes = list(self.pred_clusters[label])  # 转换为列表以便索引
            n_k = len(cluster_nodes)

            if n_k == 0:
                continue

            # 提取簇内节点的特征
            cluster_features = self.features[cluster_nodes]

            # --- 计算单个簇 C_k 的同质性 H(C_k) ---
            h_c_k = 0.0
            for j in range(p):  # 遍历每个特征维度
                # 计算簇内具有该特征（值非零）的节点数
                count_j = np.count_nonzero(cluster_features[:, j])

                # 计算比例 c_kj
                c_kj = count_j / n_k if n_k > 0 else 0.0
                # 累加项
                h_c_k += c_kj * (1 - c_kj)

            # H(C_k) 是累加项的负值
            h_c_k = -h_c_k

            # 累加到总和
            total_sum += n_k * h_c_k

        # 计算最终的归一化同质性
        denominator = N * p
        if denominator == 0:
            return 0.0

        nor_ho = total_sum / denominator
        return nor_ho

    def _calculate_L_in(self, cluster_nodes):
        """
        计算一个簇的内部边数 L_in
        """
        l_in = 0
        # 将集合转换为列表，方便遍历
        nodes_list = list(cluster_nodes)
        # 遍历簇内所有节点对 (i, j) 且 i < j，避免重复计算
        for i in range(len(nodes_list)):
            for j in range(i + 1, len(nodes_list)):
                u, v = nodes_list[i], nodes_list[j]
                # 检查 u 是否在 v 的邻居列表中
                if u in self.neighbors[v]:
                    l_in += 1
        return l_in

    def _calculate_L_out(self, cluster_nodes):
        """
        计算一个簇的外部边数 L_out
        """
        l_out = 0
        # 簇外节点集合
        all_nodes_set = set(range(self.num_nodes))
        external_nodes = all_nodes_set - cluster_nodes

        for u in cluster_nodes:
            # u 的邻居中属于簇外的节点数量
            # 使用集合的交集操作高效计算
            l_out += len(self.neighbors[u].intersection(external_nodes))

        # 每条外部边被计算了两次 (u->v 和 v->u)，因此需要除以 2
        return l_out // 2

    def normalized_tightness(self):
        """
        计算归一化紧密度 (Normalized Tightness, NorTi)

        公式: NorTi = (1 / sum(n_k)) * sum( n_k * ( 2*L_in_k/(n_k^2) - L_out_k/(n_k*(N-n_k)) ) )

        返回:
        float: 归一化紧密度的值
        """
        total_n_k = 0
        sum_terms = 0.0

        for label in self.cluster_ids:
            cluster_nodes = self.pred_clusters[label]
            n_k = len(cluster_nodes)

            # 如果簇的大小为0或1，其内部边数为0，贡献也为0，可跳过
            if n_k <= 1:
                continue

            total_n_k += n_k

            # 计算簇内边数 L_in_k
            l_in_k = self._calculate_L_in(cluster_nodes)

            # 计算簇外边数 L_out_k
            l_out_k = self._calculate_L_out(cluster_nodes)

            # 计算簇内密度项: 2*L_in_k / (n_k^2)
            term1 = (2 * l_in_k) / (n_k**2)

            # 计算簇间密度项: L_out_k / (n_k * (N - n_k))
            # N 是总节点数, (N - n_k) 是簇外节点数
            denominator_term2 = n_k * (self.num_nodes - n_k)
            term2 = l_out_k / denominator_term2 if denominator_term2 != 0 else 0.0

            # 累加各项
            sum_terms += n_k * (term1 - term2)

        # 避免除以零的情况
        if total_n_k == 0:
            return 0.0

        nor_ti = sum_terms / total_n_k
        return nor_ti

    def get_all_metrics(self):
        """
        计算并返回所有内部评估指标
        """
        _round = 4
        metrics = {
            "NorHo": round(self.normalized_homogeneity(), _round),
            "NorTi": round(self.normalized_tightness(), _round),
        }
        return metrics


class externalEvaluator:
    def __init__(self, cluster_labels, reference_labels):
        """
        聚类评估类：比较聚类结果和参考标签的外部一致性指标
        """
        self.cluster_labels = np.array(cluster_labels)

        self.reference_labels = np.array(reference_labels)

        if len(self.cluster_labels) != len(self.reference_labels):
            raise ValueError("聚类结果与参考标签长度必须一致")

        self.m = len(self.cluster_labels)
        # 预计算簇集合
        self.pred_clusters = {
            label: set(np.where(self.cluster_labels == label)[0])
            for label in np.unique(self.cluster_labels)
        }
        self.ref_clusters = {
            label: set(np.where(self.reference_labels == label)[0])
            for label in np.unique(self.reference_labels)
        }

    def _pairwise_counts(self):
        """计算 a,b,c,d 列联表参数"""
        a = b = c = d = 0
        for i, j in combinations(range(self.m), 2):
            same_cluster = self.cluster_labels[i] == self.cluster_labels[j]
            same_ref = self.reference_labels[i] == self.reference_labels[j]
            if same_cluster and same_ref:
                a += 1
            elif same_cluster and not same_ref:
                b += 1
            elif not same_cluster and same_ref:
                c += 1
            else:
                d += 1
        return a, b, c, d

    def micro_f1(self):
        """
        计算聚类外部评估的微F1分数（Micro-F1 Score）

        核心逻辑：
            1. 精确率（Precision）：聚类判定为“同簇”的样本对中，真实“同标”的比例 → a/(a+b)
            2. 召回率（Recall）：真实“同标”的样本对中，聚类判定为“同簇”的比例 → a/(a+c)
            3. 微F1：精确率与召回率的调和平均 → 2*(P*R)/(P+R)

        Returns:
            float: 微F1分数（范围0~1，分数越高，聚类与参考标签一致性越好）

        异常处理：
            若精确率和召回率均为0（无有效同簇/同标对），返回0避免除以零
        """
        # 1. 获取列联表参数
        a, b, c, _ = self._pairwise_counts()  # d不参与微F1计算，用_忽略

        # 2. 计算全局精确率（Precision）和召回率（Recall）
        # 避免除以零：若聚类无同簇对（a+b=0），精确率为0；若参考无同标对（a+c=0），召回率为0
        precision = a / (a + b) if (a + b) != 0 else 0.0
        recall = a / (a + c) if (a + c) != 0 else 0.0

        # 3. 计算微F1（调和平均）
        if precision + recall == 0:
            return 0.0  # 无有效预测时返回0
        micro_f1_score = 2 * (precision * recall) / (precision + recall)

        return micro_f1_score  # 保留4位小数，便于结果解读

    def precision(self, i, j):
        """按公式计算真实簇 i 和预测簇 j 的 Precision"""
        ref_set = self.ref_clusters[i]
        pred_set = self.pred_clusters[j]
        if len(ref_set) == 0:
            return 0
        return len(ref_set & pred_set) / len(ref_set)

    def recall(self, i, j):
        """按公式计算真实簇 i 和预测簇 j 的 Recall"""
        ref_set = self.ref_clusters[i]
        pred_set = self.pred_clusters[j]
        if len(pred_set) == 0:
            return 0
        return len(ref_set & pred_set) / len(pred_set)

    def f1_score(self):
        """返回所有簇组合的 F1-score 累加值"""
        total_f1 = 0
        for i in self.ref_clusters.keys():
            for j in self.pred_clusters.keys():
                p = self.precision(i, j)
                r = self.recall(i, j)
                if p + r > 0:
                    total_f1 += 2 * p * r / (p + r)
        return total_f1

    def accuracy(self):
        """聚类准确率 ACC，基于最优映射"""
        y_true = self.reference_labels
        y_pred = self.cluster_labels
        D = max(y_pred.max(), y_true.max()) + 1
        # 构建混淆矩阵
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1
        # 求最大匹配
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        acc = w[row_ind, col_ind].sum() / len(y_pred)
        return acc

    def jaccard_coefficient(self):
        """Jaccard 系数"""
        a, b, c, _ = self._pairwise_counts()
        return a / (a + b + c) if (a + b + c) else 0

    def fowlkes_mallows_index(self):
        """Fowlkes-Mallows 指数 (sklearn)"""
        return fowlkes_mallows_score(self.reference_labels, self.cluster_labels)

    def rand_index(self):
        """Rand 指数"""
        a, _, _, d = self._pairwise_counts()
        return (a + d) / comb(self.m, 2)

    def adjusted_rand_index(self):
        """调整后的 Rand 指数"""
        return adjusted_rand_score(self.reference_labels, self.cluster_labels)

    def normalized_mutual_information(self):
        """归一化互信息"""
        return normalized_mutual_info_score(self.reference_labels, self.cluster_labels)

    def homogeneity_completeness_vmeasure(self):
        """同质性、完整性、V-measure"""
        return homogeneity_completeness_v_measure(
            self.reference_labels, self.cluster_labels
        )

    def get_all_metrics(self):
        """返回全部指标结果"""
        h, c, v = self.homogeneity_completeness_vmeasure()
        a, b, c2, d = self._pairwise_counts()
        _round = 4
        return {
            "ACC": round(self.accuracy(), _round),  # 准确率
            "JC": round(self.jaccard_coefficient(), _round),  # Jaccard 系数
            "FMI": round(self.fowlkes_mallows_index(), _round),  # Fowlkes-Mallows 指数
            "RI": round(self.rand_index(), _round),  # Rand 指数
            "ARI": round(
                self.adjusted_rand_index(), _round
            ),  # 调整后的 Rand 指数，0表示完全随机
            "NMI": round(self.normalized_mutual_information(), _round),  # 归一化互信息
            "Micro-F1": round(self.micro_f1(), _round),
            # "SS": a,  # 预测与真实均为正的样本对数
            # "SD": b,  # 预测为正而真实为负的样本对数
            # "DS": c2,  # 预测为负而真实为正的样本对数
            # "DD": d,  # 预测与真实均为负的样本对数
        }

    def print_metrics(self):
        """打印结果"""
        metrics = self.get_all_metrics()
        print("\n=== 聚类外部指标评估结果 ===")
        for k, v in metrics.items():
            print(f"{k:25s}: {v:.4f}" if isinstance(v, float) else f"{k:25s}: {v}")


if __name__ == "__main__":
    import pandas as pd

    cluster_labels = pd.read_csv(r"results\pred_and_real\citeseer_20251111_0804.csv")[
        "predict"
    ].values
    edges = pd.read_csv(r"stgraphs\citeseer.edges", header=None, sep=",")
    features = json2featmat("citeseer").toarray()

    ie = internalEvaluator(cluster_labels, edges.values, features)
    print(ie.normalized_homogeneity())
