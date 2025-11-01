import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.evaluator import Evaluator
from src.helpers import paramsManager, dataStorageManager
from src.ml_jnmf import ML_JNMF


class Experiment:
    def __init__(self, dataname, p=2, theta=0.5, mu1=1, mu2=2, sample_size=None):
        """
        输入dataname, 开箱即用
        params: p, theta, mu1, mu2, default: p=2, theta=0.5, mu1=1, mu2=2
        """
        self.dataname = dataname
        self.experiment_time = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        self.experiment_data_dir = os.path.join(r"data\intermediate", dataname)
        self.log_file = f"results/logs/run_{self.experiment_time}.json"

        self.targets = None

        self.final_loss = None
        self.is_early_stopped = None
        self.metrices = None
        self.pred_method = None
        self.sample_size = sample_size

        self.model = ML_JNMF(mu1,mu2)
        if p is None:
            self.p = 2
        else:
            self.p = p
        if theta is None:
            self.theta = 0.5
        else:
            self.theta = theta
        if mu1 is None:
            self.mu1 = 1
        else:
            self.mu1 = mu1
        if mu2 is None:
            self.mu2 = 2
        else:
            self.mu2 = mu2

        self.storage_manager = dataStorageManager(dataname, p, theta, sample_size)

    def load_data(self):
        """加载实验数据并计算交互层矩阵

        此方法从存储管理器获取文件名，加载预计算的结构层和属性层数据，
        并基于这些数据计算交互层矩阵（包含高阶信息）。

        返回值:
            tuple: 包含三个numpy数组的元组
                - layer_structure: 结构层矩阵
                - layer_arttribute: 属性层矩阵
                - layer_inter: 交互层矩阵（融合了结构和属性信息）
        """
        # 获取要加载的文件名
        filename = self.storage_manager.file_to_save

        # 从文件中加载数据
        data = joblib.load(os.path.join(self.experiment_data_dir, filename))
        # 提取属性相似度矩阵
        layer_arttribute = data["similarity_matrix"]
        # 提取高阶结构样本矩阵并转换为数组格式
        layer_structure = data["high_order_matrix"].toarray()

        layer_arttribute /= np.max(layer_arttribute)
        layer_structure /= np.max(layer_structure)

        # 计算结构层和属性层的平均矩阵
        layer_average = (layer_structure + layer_arttribute) / 2
        # 初始化交互层为平均矩阵
        layer_inter = layer_average

        # 初始化term变量为平均矩阵（对应k=1时的项）
        term = layer_average

        # 循环计算从2到p阶的项并累加到交互层中
        for k in range(2, self.p + 1):
            # 递推公式：theta^{k-1} * layer_avg^k / k!
            term = self.theta * term @ layer_average / k
            layer_inter += term

        layer_inter /= np.max(layer_inter)

        # 加载目标标签数据
        self.targets = data["targets"]
        self.r = len(np.unique(self.targets))
        # 返回结构层、属性层和交互层矩阵
        return layer_arttribute, layer_structure, layer_inter

    def run(self, pred_method, lamb=0.5, write_log = False):
        la, ls, li = self.load_data()

        self.pred_method = pred_method
        self.cluster_labels = self.model.fit_predict(
            la, ls, li, self.r, self.pred_method, lamb
        )
        self.is_early_stopped = self.model.is_early_stopped
        self.final_loss = self.model.final_loss 
        self.evaluate()
        if write_log:
            self.write_experiment_log()
        return self

    def evaluate(self):
        cluster_labels = self.cluster_labels
        reference_labels = self.targets
        eva = Evaluator(cluster_labels, reference_labels)
        self.metrices = eva.get_all_metrics()

    def write_experiment_log(self):
        log = {
            "dataname": self.dataname,
            "r": self.r,
            "params": {
                "p": self.p,
                "theta": self.theta,
                "mu1": self.mu1,
                "mu2": self.mu2,
            },
            "time": self.experiment_time,
            "final_loss": self.final_loss,
            "is_early_stopped": self.is_early_stopped,
            "evalution": self.metrices,
            "predict_method": self.pred_method,
        }

        with open(self.log_file, "w") as f:
            json.dump(log, f, indent=4)


class testExperiment(Experiment):
    """
    测试版实验类，只加载部分数据进行快速测试
    继承自 experiment 类
    """

    def __init__(self, dataname, r, *params):
        super().__init__(dataname, r, *params)
        self.log_file = f"results/logs/test_{self.experiment_time}.json"

    # 重写 load_data 方法
    def load_data(self):
        # 只加载数据进行测试
        filename = self.storage_manager.file_to_save

        data = joblib.load(os.path.join(self.experiment_data_dir, filename))

        layer_arttribute = data["similarity_matrix"]
        layer_structure = data["high_order_sample"].toarray()
        layer_average = (layer_structure + layer_arttribute) / 2
        layer_inter = layer_average

        for k in range(2, self.p + 1):
            term = (
                self.theta * term @ layer_average / k
            )  # 递推：theta ^ {k-1} * layer_avg^k / k!
            layer_inter += term

        self.targets = np.load(self.targets_path)[:100]

        return (
            layer_structure[:100, :100],
            layer_arttribute[:100, :100],
            layer_inter[:100, :100],
        )


class searchExperiment:
    def __init__(self, dataname, sample_size):
        self.dataname = dataname
        self.output = f"results/search/{dataname}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        os.makedirs("results/search", exist_ok=True)
        self.sample_size = sample_size

    def run(self, pred_method, overwrite=False):
        pm = paramsManager()
        if not overwrite:
            try:
                df = pd.read_csv(self.output)
                existing_params = set(zip(df["p"], df["theta"], df["mu1"], df["mu2"]))
            except FileNotFoundError:
                existing_params = set()
        else:
            existing_params = set()
        df = pd.DataFrame(
            columns=[
                "p",
                "theta",
                "mu1",
                "mu2",
                "final_loss",
                "is_early_stopped",
                "predict_method",
                "ACC",  # 准确率
                "JC",  # Jaccard 系数
                "FMI",  # Fowlkes-Mallows 指数
                "RI",  # Rand 指数
                "ARI",  # 调整后的 Rand 指数，0表示完全随机
                "NMI",  # 归一化互信息
                "HOMO",  # 同质性
                "COMP",  # 完整性
                "VM",  # V-measure
                "F1",  # F1 分数
                "SS",  # 预测与真实均为正的样本对数
                "SD",  # 预测为正而真实为负的样本对数
                "DS",  # 预测为负而真实为正的样本对数
                "DD",  # 预测与真实均为负的样本对数
            ]
        )
        df.to_csv(self.output, index=False)
        for mu1 in pm.mu1_to_select:
            for mu2 in pm.mu2_to_select:
                for p in pm.p_to_select:
                    for theta in pm.theta_to_select:
                        if (p, theta, mu1, mu2) in existing_params:
                            print(
                                f"跳过已存在组合: p={p}, theta={theta}, mu1={mu1}, mu2={mu2}",
                            )
                            continue

                        epr = Experiment(self.dataname, p, theta, mu1, mu2, self.sample_size)

                        epr.run(pred_method)

                        res = {
                            "p": epr.p,
                            "theta": epr.theta,
                            "mu1": epr.mu1,
                            "mu2": epr.mu2,
                            "final_loss": epr.final_loss,
                            "is_early_stopped": epr.is_early_stopped,
                            "predict_method": pred_method,
                        }
                        res.update(epr.metrices)
                        
                        df_new = pd.DataFrame([res])
                        df_new.to_csv(self.output, mode="a", header=False, index=False)
