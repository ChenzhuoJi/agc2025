import time
from typing import Literal, Union

import numpy as np
import pandas as pd
from sklearn.decomposition._nmf import _initialize_nmf
from rich.console import Console

from src.helpers import compute_communitude_metric, create_mapping


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, max_loss=1e10):
        self.patience = patience
        self.min_delta = min_delta
        self.max_loss = max_loss  # 添加最大损失值限制
        self.counter = 0

    def step(self, current_loss, best_loss):
        # 检查当前损失值是否超过最大容忍值
        if current_loss >= self.max_loss:
            return True

        # 原有的早停逻辑
        if current_loss < best_loss - self.min_delta:
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def reset(self):
        self.counter = 0


class ConvergenceChecker:
    def __init__(self, patience=10, tol=1e-4):
        self.patience = patience  # 观察最近多少次 loss
        self.tol = tol  # 判定收敛的阈值
        self.history = []  # 保存最近的 loss
        self.is_converged = False  # 是否已经收敛

    def step(self, current_loss):
        self.history.append(current_loss)
        if len(self.history) > self.patience:
            self.history.pop(0)

        # 判定收敛：最近 patience 次 loss 变化都小于 tol
        if len(self.history) == self.patience:
            diffs = [
                abs(self.history[i] - self.history[i - 1])
                / (self.history[i - 1] + 1e-10)
                for i in range(1, len(self.history))
            ]
            self.is_converged = all(diff < self.tol for diff in diffs)

        return self.is_converged

    def reset(self):
        self.history = []
        self.is_converged = False


class lossTracker:
    def __init__(self):
        """
        初始化损失跟踪器，创建用于记录各类损失值的历史记录字典

        初始化一个包含多个损失类型的字典，每个损失类型对应一个空列表，
        用于在训练过程中逐步存储各类损失值，方便后续分析和可视化
        """
        self.history = {
            "total_loss": [],  # 总损失值记录
            "atrributes_layer_loss": [],  # 属性层损失值记录
            "structure_layer_loss": [],  # 结构层损失值记录
            "inter_layer_loss": [],  # 层间损失值记录
            "pairwise_similarity_loss": [],  # 成对相似度损失值记录
            "intra_layer_loss": [],  # 层内损失值记录
        }

    def step(
        self,
        current_total_loss,
        current_attributes_layer_loss,
        current_structure_layer_loss,
        current_intra_loss,
        current_inter_loss,
        current_sim_loss,
    ):
        """
        记录训练过程中的各类损失值

        将当前训练步骤中的各类损失值添加到对应的历史记录列表中，
        用于后续监控训练过程、分析模型收敛情况和可视化损失曲线

        参数:
            current_total_loss: 当前步骤的总损失值
            current_attributes_layer_loss: 当前步骤的属性层损失值
            current_structure_layer_loss: 当前步骤的结构层损失值
            current_intra_loss: 当前步骤的层内损失值
            current_inter_loss: 当前步骤的层间损失值
            current_sim_loss: 当前步骤的成对相似度损失值
        """
        self.history["total_loss"].append(current_total_loss)
        self.history["atrributes_layer_loss"].append(current_attributes_layer_loss)
        self.history["structure_layer_loss"].append(current_structure_layer_loss)
        self.history["inter_layer_loss"].append(current_inter_loss)
        self.history["intra_layer_loss"].append(current_intra_loss)
        self.history["pairwise_similarity_loss"].append(current_sim_loss)

    def reset(self):
        self.history = {
            "total_loss": [],
            "atrributes_layer_loss": [],
            "structure_layer_loss": [],
            "intra_layer_loss": [],
            "inter_layer_loss": [],
            "pairwise_similarity_loss": [],
        }

    def plot_loss(self, deleted=[]):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 7))
        # del self.history["atrributes_layer_loss"]
        # del self.history["structure_layer_loss"]
        for key, values in self.history.items():
            if key not in deleted:
                plt.plot(values, label=key, linestyle="-", linewidth=2)

        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()

    def to_dataframe(self):
        return pd.DataFrame(self.history)

    def to_csv(self, filename="loss_history.csv"):
        self.to_dataframe().to_csv(filename, index=False)


class ML_JNMF:
    def __init__(
        self,
        la: np.ndarray,
        ls: np.ndarray,
        li: np.ndarray,
        interWeight: float = 5.0,
        pairwiseWeight: float = 2.0,
        max_iter: int = 100,
        convergence_tol: float = 1e-4,
        convergence_patience: int = 10,
        early_stopping_patience: int = 20,
        early_stopping_min_delta: float = 1e-4,
        random_state: int = 42,
    ):
        """
        初始化多模态联合非负矩阵分解(ML-JNMF)模型

        参数:
            la: np.ndarray - 属性层的高阶矩阵，表示节点属性关系的高阶表示
            ls: np.ndarray - 结构层的高阶矩阵，表示节点连接关系的高阶表示
            li: np.ndarray - 层间关联矩阵，表示属性层与结构层之间的关联关系
            interWeight: float - 层间损失的权重系数，默认为5.0
            pairwiseWeight: float - 成对相似度损失的权重系数，默认为2.0
            max_iter: int - 最大迭代次数，默认为100
            convergence_tol: float - 收敛阈值，默认为1e-4
            convergence_patience: int - 收敛检查的耐心值，默认为10
            early_stopping_patience: int - 早停机制的耐心值，默认为20
            early_stopping_min_delta: float - 早停机制的最小变化量，默认为1e-4
            random_state: int - 随机数种子，用于结果可复现，默认为42
        """
        self.mu1 = interWeight  # 层间损失的权重系数
        self.mu2 = pairwiseWeight  # 成对相似度损失的权重系数
        self.max_iter = max_iter  # 最大迭代次数
        self.random_state = random_state  # 随机数种子

        # 初始化辅助工具类
        self.early_stopper = EarlyStopping(
            patience=early_stopping_patience, min_delta=early_stopping_min_delta
        )  # 早停机制实例
        self.convergence_checker = ConvergenceChecker(
            patience=convergence_patience, tol=convergence_tol
        )  # 收敛检查器实例
        self.loss_tracker = lossTracker()  # 损失跟踪器实例

        # 保存输入的矩阵数据
        self.la = la  # 属性层的高阶矩阵
        self.ls = ls  # 结构层的高阶矩阵
        self.li = li  # 层间关联矩阵
        self.size = la.shape[0]  # 获取节点数量

        # 初始化模型结果相关变量
        self.U1, self.U2, self.B1, self.B2, self.S12 = [None] * 5  # 分解后的矩阵变量
        self.loss_history = []  # 损失历史记录列表
        self.final_loss = None  # 最终损失值
        self.is_converged = False  # 模型是否收敛的标志
        self.early_stopping = False  # 是否触发早停机制的标志
        self.community = None  # 社区划分结果

    def matrixInit(
        self, r: int, init: str = "nndsvdar"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        初始化多模态联合非负矩阵分解所需的所有矩阵

        参数:
            r: int - 分解后的低维空间维度
            init: str - 初始化方法，默认为"nndsvdar"（一种非负矩阵分解的初始化算法）

        返回值:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] - 包含初始化后的5个矩阵的元组
                U1: 属性层的基础矩阵
                U2: 结构层的基础矩阵
                B1: 层间变换矩阵1
                B2: 层间变换矩阵2
                S12: 正交约束矩阵（初始化为单位矩阵）
        """
        # 使用NMF方法初始化属性层基础矩阵U1
        self.U1, _ = _initialize_nmf(
            self.la, n_components=r, init=init, random_state=self.random_state
        )

        # 使用NMF方法初始化结构层基础矩阵U2
        self.U2, _ = _initialize_nmf(
            self.ls, n_components=r, init=init, random_state=self.random_state
        )

        # 使用NMF方法初始化层间变换矩阵B1和B2t
        # B2是B2t的转置，这样B1和B2在维度上能够匹配
        self.B1, B2t = _initialize_nmf(
            self.li, n_components=r, init=init, random_state=self.random_state
        )
        self.B2 = B2t.T

        # 处理矩阵中的NaN值，将其替换为很小的正数1e-6
        # 这是为了避免在后续迭代计算中出现数值不稳定的问题
        self.U1 = np.nan_to_num(self.U1, nan=1e-6)
        self.U2 = np.nan_to_num(self.U2, nan=1e-6)
        self.B1 = np.nan_to_num(self.B1, nan=1e-6)
        self.B2 = np.nan_to_num(self.B2, nan=1e-6)

        # 初始化正交约束矩阵S12为单位矩阵
        # 单位矩阵意味着初始时假设两个模态之间的表示是正交的
        self.S12 = np.eye(r)

        # 返回所有初始化好的矩阵
        return self.U1, self.U2, self.B1, self.B2, self.S12

    def calculateLoss(self) -> float:
        """计算多模态联合非负矩阵分解(ML-JNMF)模型的总目标函数损失值。

        该方法计算模型的综合损失，包括内部层损失、层间损失和成对相似度损失，
        并通过损失跟踪器记录当前训练步骤的各项损失值，同时进行异常值检测。

        Returns:
            float: 模型的总损失值，即所有损失项的加权和
        """

        def l21_norm(X):
            """计算矩阵的L2,1范数（行L2范数的和）"""
            return np.sum(np.sqrt(np.sum(X**2, axis=1)))

        # -------- 内部层损失（使用L2,1范数，无平方） --------
        # 属性层重构损失：衡量属性矩阵与分解后矩阵乘积的差异
        attributes_layer_loss = l21_norm(self.la - self.U1 @ self.U1.T)  # 实验分析部分
        # 结构层重构损失：衡量结构矩阵与分解后矩阵乘积的差异
        structure_layer_loss = l21_norm(self.ls - self.U2 @ self.U2.T)

        # 内部层总损失：属性层损失与结构层损失之和
        intra_loss = attributes_layer_loss + structure_layer_loss

        # -------- 层间损失（使用L2,1范数，无平方） --------
        # 衡量跨层信息传递的误差，由超参数mu1控制权重
        inter_loss = self.mu1 * l21_norm(self.li - self.B1 @ self.S12 @ self.B2.T)

        # -------- 成对相似度损失（Frobenius范数的平方） --------
        # 衡量不同表示空间之间的一致性，由超参数mu2控制权重
        sim_loss = self.mu2 * (
            np.linalg.norm(self.U1 @ self.U1.T - self.B1 @ self.B1.T, "fro") ** 2
            + np.linalg.norm(self.U2 @ self.U2.T - self.B2 @ self.B2.T, "fro") ** 2
        )

        # -------- 总损失 --------
        # 综合所有损失项，形成最终的优化目标
        total_loss = intra_loss + inter_loss + sim_loss

        # 使用损失跟踪器记录当前步骤的各项损失值，用于后续分析
        self.loss_tracker.step(
            total_loss,
            attributes_layer_loss,
            structure_layer_loss,
            intra_loss,
            inter_loss,
            sim_loss,
        )  # 暂时分析的代码

        # -------- 检查异常值（nndsvd初始化遇见全0列，即死列，会导致初始化的矩阵有nan值） --------
        if np.isnan(total_loss) or np.isinf(total_loss):
            print("⚠️ Loss NaN detected")
            print(
                "self.la:",
                np.isnan(self.la).any(),
                "A2:",
                np.isnan(self.ls).any(),
                "A12:",
                np.isnan(self.li).any(),
            )
            print("U1:", np.isnan(self.U1).any(), "U2:", np.isnan(self.U2).any())
            print(
                "B1:",
                np.isnan(self.B1).any(),
                "B2:",
                np.isnan(self.B2).any(),
                "S12:",
                np.isnan(self.S12).any(),
            )
            print(
                "Any Inf:",
                np.isinf(self.U1).any()
                or np.isinf(self.U2).any()
                or np.isinf(self.B1).any(),
            )
            raise ValueError("NaN detected in loss computation")

        return total_loss

    def multiplicativeUpdate(
        self, eps=1e-10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """更新U1, U2, B1, B2, S12矩阵，使用乘法更新规则。

        该方法根据当前的损失函数，通过矩阵乘法更新所有参数，确保非负性约束。
        更新规则考虑了内部层和层间损失，以及成对相似度损失。

        参数:
            eps (float, 可选): 小值，用于防止除零错误，默认值为1e-10。

        返回值:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                更新后的U1, U2, B1, B2, S12矩阵。
        """

        def build_Z(A, U):
            """构建对角矩阵 Z = diag(1 / ||A - UUᵀ||₂)，用于加权重构误差

            参数:
                A: 原始邻接矩阵
                U: 潜在特征矩阵

            返回值:
                numpy.ndarray: 对角权重矩阵
            """
            # 计算残差矩阵（原始矩阵与重构矩阵的差）
            residual = A - U @ U.T
            # 计算每行的L2范数，并添加小值防止除零
            norms = np.linalg.norm(residual, axis=1) + eps
            # 构建对角矩阵，对角线元素为范数的倒数
            return np.diag(1 / norms)

        def build_Z12(A12, B1, B2, S12):
            """为层间重构构建对角矩阵

            参数:
                A12: 层间连接的邻接矩阵
                B1: 第一层网络的共享嵌入矩阵
                B2: 第二层网络的共享嵌入矩阵
                S12: 层间映射矩阵

            返回值:
                numpy.ndarray: 对角权重矩阵
            """
            # 计算层间残差矩阵
            residual = A12 - B1 @ S12 @ B2.T
            # 计算每行的L2范数，并添加小值防止除零
            norms = np.linalg.norm(residual, axis=1) + eps
            # 构建对角矩阵，对角线元素为范数的倒数
            return np.diag(1 / norms)

        # 构建三层的权重矩阵Z1、Z2和Z12
        Z1, Z2, Z12 = (
            build_Z(self.la, self.U1),
            build_Z(self.ls, self.U2),
            build_Z12(self.li, self.B1, self.B2, self.S12),
        )

        # 更新U1矩阵：使用乘法更新规则，确保非负性
        # 分子部分：包含重构误差和与B1的一致性约束
        U1_num = (
            Z1 @ self.la @ self.U1
            + self.la @ Z1 @ self.U1
            + 2 * self.mu2 * self.B1 @ self.B1.T @ self.U1
        ) * self.U1
        # 分母部分：包含归一化项
        U1_den = (
            self.U1 @ self.U1.T @ Z1 @ self.U1
            + Z1 @ self.U1 @ self.U1.T @ self.U1
            + 2 * self.mu2 * self.U1 @ self.U1.T @ self.U1
            + eps
        )
        # 执行更新
        self.U1 = U1_num / U1_den

        # 更新U2矩阵：与U1类似的更新规则
        U2_num = (
            Z2 @ self.ls @ self.U2
            + self.ls @ Z2 @ self.U2
            + 2 * self.mu2 * self.B2 @ self.B2.T @ self.U2
        ) * self.U2
        U2_den = (
            self.U2 @ self.U2.T @ Z2 @ self.U2
            + Z2 @ self.U2 @ self.U2.T @ self.U2
            + 2 * self.mu2 * self.U2 @ self.U2.T @ self.U2
            + eps
        )
        self.U2 = U2_num / U2_den

        # 更新B1矩阵：结合层间重构和与U1的一致性约束
        B1_num = (
            self.mu1 * self.li @ Z12 @ self.B2 @ self.S12.T
            + 2 * self.mu2 * self.U1 @ self.U1.T @ self.B1
        ) * self.B1
        B1_den = (
            self.mu1 * self.B1 @ self.S12 @ self.B2.T @ Z12 @ self.B2 @ self.S12.T
            + 2 * self.mu2 * self.B1 @ self.B1.T @ self.B1
            + eps
        )
        self.B1 = B1_num / B1_den

        # 更新B2矩阵：与B1类似的更新规则
        B2_num = (
            self.mu1 * Z12 @ self.li.T @ self.B1 @ self.S12
            + 2 * self.mu2 * self.U2 @ self.U2.T @ self.B2
        ) * self.B2
        B2_den = (
            self.mu1 * Z12 @ self.B2 @ self.S12.T @ self.B1.T @ self.B1 @ self.S12
            + 2 * self.mu2 * self.B2 @ self.B2.T @ self.B2
            + eps
        )
        self.B2 = B2_num / B2_den

        # 更新S12矩阵：层间映射矩阵的更新规则
        S12_num = self.B1.T @ self.li @ Z12 @ self.B2
        S12_den = self.B1.T @ self.B1 @ self.S12 @ self.B2.T @ Z12 @ self.B2 + eps
        self.S12 = (S12_num / S12_den) * self.S12

        # 返回所有更新后的矩阵
        return self.U1, self.U2, self.B1, self.B2, self.S12

    def fit(self, r: int, silent: bool = True) -> None:
        """
        训练ML-JNMF模型，使用EarlyStopping和ConvergenceChecker进行监控。

        参数:
            r (int): 潜在特征维度，即社区数量。
            silence (bool, 可选): 是否在训练过程中静默打印日志，默认值为True。
        """
        console = Console()
        t_fit = time.time()
        # Initialize matrices
        self.U1, self.U2, self.B1, self.B2, self.S12 = self.matrixInit(r)

        # 初始化训练管理器
        best_loss = float("inf")
        best_params = None

        self.early_stopper.reset()
        self.convergence_checker.reset()
        self.loss_tracker.reset()

        for it in range(self.max_iter):
            # 计算当前 loss
            time_start = time.time()
            loss = self.calculateLoss()
            self.loss_history.append(loss)
            if self.early_stopper.step(loss, best_loss):
                # console.print(f"loss={loss:.4f}, best_loss={best_loss:.4f}")
                self.U1, self.U2, self.B1, self.B2, self.S12 = best_params
                self.is_converged = False
                self.early_stopping = True
                self.final_loss = best_loss
                t_fit = time.time() - t_fit
                console.print(
                    f"[Early Stop] iteration={it+1}, best_loss={best_loss:.4f} at iteration {best_it+1}, n_nodes={self.size}, computing_time={t_fit:.4f}s",
                    style="bold yellow",
                )
                break

            if loss < best_loss:
                best_loss = loss
                best_params = (
                    self.U1.copy(),
                    self.U2.copy(),
                    self.B1.copy(),
                    self.B2.copy(),
                    self.S12.copy(),
                )
                best_it = it

            # 收敛性检查
            if self.convergence_checker.step(loss):
                self.U1, self.U2, self.B1, self.B2, self.S12 = best_params
                self.final_loss = best_loss
                self.is_converged = True
                t_fit = time.time() - t_fit
                console.print(
                    f"[Converged] iteration={it+1}, loss={loss:.4f}, computing_time={t_fit:.4f}s",
                    style="bold green",
                )
                break

            # 更新因子矩阵
            self.U1, self.U2, self.B1, self.B2, self.S12 = self.multiplicativeUpdate()
            if not silent:
                if it % 5 == 0:
                    console.print(
                        f"[Update] iteration={it+1}, loss={loss:.4f}, best_loss={best_loss:.4f}, computing_time={time.time() - time_start:.4f}s",
                        style="blue",
                    )
                
        # 如果循环自然结束，也使用最佳参数
        else:
            self.U1, self.U2, self.B1, self.B2, self.S12 = best_params
            self.final_loss = best_loss
            t_fit = time.time() - t_fit
            console.print(
                f"[End] iteration={it+1}, loss={best_loss:.4f}, computing_time={t_fit:.4f}s",
                style="bold purple",
            )
        return self

    def predict(
        self, r: int, pred_method: Literal["lambda", "communitude"], lamb: float = 0.5
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        预测社区结构，根据选择的方法返回不同的结果。

        参数:
            r (int): 潜在特征维度，即社区数量。
            pred_method (Literal["lambda", "communitude"]): 预测方法，可选值包括 'lambda' 和 'communitude'。
            lamb (float, 可选): 用于加权组合U1和U2的参数，范围在[0,1]之间，默认值为0.5。

        返回值:
            Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
                如果 pred_method 为 'lambda'，则返回一个数组，包含每个节点的社区标签。
                如果 pred_method 为 'communitude'，则返回一个元组，包含两个数组，分别为每个节点的内部层和层间社区标签。
        """
        if pred_method == "lambda":
            Z = lamb * self.U1 + (1 - lamb) * self.U2
            return np.argmax(Z, axis=1)

        elif pred_method == "communitude":
            # -------- Obtain intra-layer and inter-layer community labels --------
            label_intra_1 = np.argmax(self.U1, axis=1)
            label_inter_1 = np.argmax(self.B1, axis=1)
            label_intra_2 = np.argmax(self.U2, axis=1)
            label_inter_2 = np.argmax(self.B2, axis=1)

            # -------- Calculate community metrics --------
            comm_intra_1 = compute_communitude_metric(self.la, label_intra_1)
            comm_intra_2 = compute_communitude_metric(self.ls, label_intra_2)
            comm_inter_1 = compute_communitude_metric(self.li, label_inter_1, axis=0)
            comm_inter_2 = compute_communitude_metric(self.li, label_inter_2, axis=1)

            # -------- Determine final community type based on metrics --------
            final_community_1 = []
            final_community_2 = []

            for i in range(len(label_intra_1)):
                if comm_inter_1[label_inter_1[i]] > comm_intra_1[label_intra_1[i]]:
                    final_community_1.append(("inter", label_inter_1[i]))
                else:
                    final_community_1.append(("intra", label_intra_1[i]))

            for j in range(len(label_intra_2)):
                if comm_inter_2[label_inter_2[j]] > comm_intra_2[label_intra_2[j]]:
                    final_community_2.append(("inter", label_inter_2[j]))
                else:
                    final_community_2.append(("intra", label_intra_2[j]))

            df1 = pd.DataFrame(final_community_1, columns=["type", "community_id"])
            df1.insert(0, "node_id", range(len(final_community_1)))
            df2 = pd.DataFrame(final_community_2, columns=["type", "community_id"])
            df2.insert(0, "node_id", range(len(final_community_2)))
            df1["community_id"] = df1.apply(create_mapping, axis=1)
            df2["community_id"] = df2.apply(create_mapping, axis=1)
            return df1["community_id"], df2["community_id"]

    def fit_predict(
        self,
        r: int,
        pred_method: Literal["lambda", "communitude"],
        lamb: float = 0.5,
        silent: bool = True,
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        训练ML-JNMF模型并预测社区结构，根据选择的方法返回不同的结果。

        参数:
            r (int): 潜在特征维度，即社区数量。
            pred_method (Literal["lambda", "communitude"]): 预测方法，可选值包括 'lambda' 和 'communitude'。
            lamb (float, 可选): 用于加权组合U1和U2的参数，范围在[0,1]之间，默认值为None。
            silent (bool, 可选): 是否在训练过程中静默打印日志，默认值为True。

        返回值:
            Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
                如果 pred_method 为 'lambda'，则返回一个数组，包含每个节点的社区标签。
                如果 pred_method 为 'communitude'，则返回一个元组，包含两个数组，分别为每个节点的内部层和层间社区标签。
        """
        self.fit(r, silent)
        self.community = self.predict(r, pred_method, lamb)
        return self.community