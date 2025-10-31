"""
core/metrics_simplified.py

Simplified and optimized version of external clustering evaluation metrics.
"""

import numpy as np
from itertools import combinations
from scipy.special import comb
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    jaccard_score,
    homogeneity_completeness_v_measure
)
from scipy.optimize import linear_sum_assignment

class Evaluator:
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
        self.pred_clusters = {label: set(np.where(self.cluster_labels == label)[0])
                              for label in np.unique(self.cluster_labels)}
        self.ref_clusters = {label: set(np.where(self.reference_labels == label)[0])
                             for label in np.unique(self.reference_labels)}
    def _pairwise_counts(self):
        """计算 a,b,c,d 列联表参数"""
        a = b = c = d = 0
        for i, j in combinations(range(self.m), 2):
            same_cluster = (self.cluster_labels[i] == self.cluster_labels[j])
            same_ref = (self.reference_labels[i] == self.reference_labels[j])
            if same_cluster and same_ref:
                a += 1
            elif same_cluster and not same_ref:
                b += 1
            elif not same_cluster and same_ref:
                c += 1
            else:
                d += 1
        return a, b, c, d

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
        return homogeneity_completeness_v_measure(self.reference_labels, self.cluster_labels)

    def get_all_metrics(self):
        """返回全部指标结果"""
        h, c, v = self.homogeneity_completeness_vmeasure()
        a, b, c2, d = self._pairwise_counts()
        return {
            "Accuracy": self.accuracy(),
            "Jaccard_Coefficient": self.jaccard_coefficient(),
            "Fowlkes_Mallows_Index": self.fowlkes_mallows_index(),
            "Rand_Index": self.rand_index(),
            "Adjusted_Rand_Index": self.adjusted_rand_index(),
            "Normalized_MI": self.normalized_mutual_information(),
            "Homogeneity": h,
            "Completeness": c,
            "V-Measure": v,
            "F1_Score": self.f1_score(),
            "SS": a, "SD": b, "DS": c2, "DD": d
        }
    def print_metrics(self):
        """打印结果"""
        metrics = self.get_all_metrics()
        print("\n=== 聚类外部指标评估结果 ===")
        for k, v in metrics.items():
            print(f"{k:25s}: {v:.4f}" if isinstance(v, float) else f"{k:25s}: {v}")
