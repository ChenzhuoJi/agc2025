"""
Author: Pumpkin🎃
Date:2025-10-28
Description:通用辅助函数
"""

import os
import json
import joblib
import numpy as np

class paramsManager:
    """管理超参数: configs/hparams/hparams_search.json"""

    def __init__(self):
        with open("configs/hparams/hparams_search.json", "r") as f:
            self.hparams = json.load(f)["hyperparameters"]
            self.p_to_select = self.hparams["p"]
            self.theta_to_select = self.hparams["theta"]
            self.mu1_to_select = self.hparams["mu1"]
            self.mu2_to_select = self.hparams["mu2"]


class dataStorageManager:
    """管理数据存储与加载的工具类"""

    def __init__(self, dataname, p=2, theta=0.5, sample_size=None):
        """
        初始化存储管理器

        Args:
            dataname (str): 数据集名称
            p (float): 参数 p
            theta (float): 参数 theta
            sample_size (int): 样本大小
        """
        self.dataname = dataname
        self._params = (p, theta, sample_size)

        # 初始化参数管理器（用于确定步长或索引）
        pm = paramsManager()
        self.theta_to_select = pm.theta_to_select
        self.p_to_select = pm.p_to_select

        # 创建目录
        self._build_dir()
        self.raw_dir, self.intermediate_dir = self._get_dir()

    # ------------------------------------------------------------------
    # 📦 参数属性
    # ------------------------------------------------------------------
    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, new_params):
        if not (isinstance(new_params, (list, tuple)) and len(new_params) in [2,3]):
            raise ValueError("params 应为 (p, theta) 或 (p, theta, sample_size)")
        self._params = tuple(new_params)

    # ------------------------------------------------------------------
    # 📁 路径与文件命名
    # ------------------------------------------------------------------
    def _build_dir(self):
        """创建原始与中间数据目录"""
        os.makedirs(os.path.join("data/raw", self.dataname), exist_ok=True)
        os.makedirs(os.path.join("data/intermediate", self.dataname), exist_ok=True)

    def _get_dir(self):
        """返回原始与中间数据目录"""
        raw_dir = os.path.join("data/raw", self.dataname)
        inter_dir = os.path.join("data/intermediate", self.dataname)
        return raw_dir, inter_dir

    def _auto_format(self, x, step=0.01):
        """根据步长自动控制小数精度"""
        decimals = abs(int(np.floor(np.log10(step)))) if step > 0 else 3
        return f"{x:.{decimals}f}".replace(".", "_")

    @property
    def file_to_save(self):
        """根据参数自动生成文件名（小数点转下划线）"""
        p, theta, sample_size = self._params

        # 根据 paramsManager 自动推断步长
        if len(self.p_to_select) > 1:
            p_step = abs(self.p_to_select[1] - self.p_to_select[0])
        else:
            p_step = 0.01

        if len(self.theta_to_select) > 1:
            theta_step = abs(self.theta_to_select[1] - self.theta_to_select[0])
        else:
            theta_step = 0.01

        p_str = self._auto_format(p, p_step)
        theta_str = self._auto_format(theta, theta_step)
        sample_size_str = f"{sample_size}" if sample_size is not None else ""
        return f"{sample_size_str}p{p_str}_theta{theta_str}.pkl"

    @property
    def output(self):
        """输出文件完整路径"""
        return os.path.join(self.intermediate_dir, self.file_to_save)

    # ------------------------------------------------------------------
    # 💾 文件操作
    # ------------------------------------------------------------------
    def exists(self):
        """检查当前参数对应的文件是否存在"""
        return os.path.exists(self.output)

    def save(self, data, overwrite=False):
        """
        保存数据文件（默认不覆盖）

        Args:
            data: 要保存的对象
            overwrite (bool): 是否允许覆盖已存在文件
        """
        if not overwrite and self.exists():
            print(f"[Skip] File already exists: {self.output}")
            return

        joblib.dump(data, self.output, compress=("gzip", 3))
        print(f"[Saved] {self.output}")

    def load(self):
        """加载已保存的数据"""
        if not self.exists():
            raise FileNotFoundError(f"{self.output} 不存在")
        return joblib.load(self.output)
    
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
        denominator = (
            (e_intra_ck + e_inter_ck) / (2 * total_edge_weight)
        ) ** 2 * (
            1 - ((e_intra_ck + e_inter_ck) / (2 * total_edge_weight)) ** 2
        )

        results[ck] = 0.0 if denominator == 0 else numerator / denominator

    return results

def create_mapping(row):
    if row["type"] == "intra":
        return 100 + row["community_id"]
    else:
        return 200 + row["community_id"]
    
