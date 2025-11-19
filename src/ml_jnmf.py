import time
from typing import Literal, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition._nmf import _initialize_nmf
from rich.console import Console
import torch

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
        ls: np.ndarray, # 不能用稀疏矩阵，用稀疏矩阵爆内存
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

        # ----------------------------
        # 1. 检测 GPU 是否可用
        # ----------------------------
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.device = torch.device("cuda")
            print(">>> GPU detected, using PyTorch + CUDA")
        else:
            self.device = torch.device("cpu")
            print(">>> No GPU found, using CPU numpy version")

        # ----------------------------
        # 2. 保存初始化参数
        # ----------------------------
        self.mu1 = interWeight
        self.mu2 = pairwiseWeight
        self.max_iter = max_iter
        self.random_state = random_state

        # EarlyStopping / ConvergenceChecker
        self.early_stopper = EarlyStopping(
            patience=early_stopping_patience, min_delta=early_stopping_min_delta
        )
        self.convergence_checker = ConvergenceChecker(
            patience=convergence_patience, tol=convergence_tol
        )
        self.loss_tracker = lossTracker()

        # ----------------------------
        # 3. 根据运行模式转为 numpy or torch
        # ----------------------------

        if self.use_gpu:
            # ---- GPU 模式 -> 转成 torch.Tensor 并放到 CUDA ----
            self.la = torch.tensor(la, dtype=torch.float32, device=self.device)
            if sp.issparse(ls):
                self.ls = torch.tensor(
                    ls.toarray(), dtype=torch.float32, device=self.device
                )
            else:
                self.ls = torch.tensor(ls, dtype=torch.float32, device=self.device)
            self.li = torch.tensor(li, dtype=torch.float32, device=self.device)

        else:
            # ---- CPU 模式 -> 保持 numpy + scipy ----
            self.la = la
            self.ls = ls
            self.li = li

        self.size = la.shape[0]

        # ----------------------------
        # 4. 初始化模型变量
        # ----------------------------
        self.U1 = None
        self.U2 = None
        self.B1 = None
        self.B2 = None
        self.S12 = None

        self.loss_history = []
        self.final_loss = None
        self.is_converged = False
        self.early_stopping = False
        self.community = None

    def matrixInit(self, r: int, init: str = "nndsvdar"):
        """
        初始化多模态联合非负矩阵分解所需的所有矩阵（GPU/CPU 自动支持）
        """

        # --------------------------
        # 1. NMF 初始化全部在 CPU/numpy 上运行
        # --------------------------
        U1, _ = _initialize_nmf(
            self.la, n_components=r, init=init, random_state=self.random_state
        )
        U2, _ = _initialize_nmf(
            self.ls, n_components=r, init=init, random_state=self.random_state
        )
        B1, B2t = _initialize_nmf(
            self.li, n_components=r, init=init, random_state=self.random_state
        )
        B2 = B2t.T

        # 替换 NaN
        U1 = np.nan_to_num(U1, nan=1e-6)
        U2 = np.nan_to_num(U2, nan=1e-6)
        B1 = np.nan_to_num(B1, nan=1e-6)
        B2 = np.nan_to_num(B2, nan=1e-6)

        S12 = np.eye(r)

        # 先保存 numpy 版本（CPU 模式直接用）
        self.U1, self.U2, self.B1, self.B2, self.S12 = U1, U2, B1, B2, S12

        # 如果没有 GPU，就返回 numpy
        if not self.use_gpu:
            return self.U1, self.U2, self.B1, self.B2, self.S12

        # --------------------------
        # 2. GPU 模式：转换为 torch
        # --------------------------
        device = torch.device("cuda")

        # la / ls / li：输入矩阵本身也要转成 torch（ls 是 csr 稀疏，需要 .toarray()）
        self.la = torch.tensor(self.la, dtype=torch.float32, device=device)
        self.ls = torch.tensor(self.ls.toarray(), dtype=torch.float32, device=device)
        self.li = torch.tensor(self.li, dtype=torch.float32, device=device)

        # 将初始化参数矩阵送上 GPU
        self.U1 = torch.tensor(U1, dtype=torch.float32, device=device)
        self.U2 = torch.tensor(U2, dtype=torch.float32, device=device)
        self.B1 = torch.tensor(B1, dtype=torch.float32, device=device)
        self.B2 = torch.tensor(B2, dtype=torch.float32, device=device)
        self.S12 = torch.tensor(S12, dtype=torch.float32, device=device)

        return self.U1, self.U2, self.B1, self.B2, self.S12

    def calculateLoss(self) -> float:
        """Torch + NumPy 混合兼容的 ML-JNMF 损失计算"""

        use_torch = self.use_gpu  # 是否使用 torch
        device = self.device if self.use_gpu else None

        # ============= 公共工具函数： L21 Norm ============= #
        def l21_norm_torch(X):
            return torch.norm(X, dim=1).sum()

        def l21_norm_np(X):
            return np.sum(np.sqrt(np.sum(X**2, axis=1)))

        # ============= 1. Intra-layer Loss (L21) ============= #
        def efficient_l21_norm_intra(A, U, batch_size=1000):
            n = A.shape[0]
            total = 0.0

            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)

                if use_torch:
                    A_batch = A[i:end]  # (bs, n)
                    U_batch = U[i:end]  # (bs, r)
                    recon = U_batch @ U.T  # (bs, n)
                    total += l21_norm_torch(A_batch - recon)
                else:
                    if sp.issparse(A):
                        A_batch = A[i:end].toarray()
                    else:
                        A_batch = A[i:end]
                    U_batch = U[i:end]
                    recon = U_batch @ U.T
                    total += l21_norm_np(A_batch - recon)

            return total

        # ============= 2. Inter-layer Loss ============= #
        def efficient_l21_norm_inter(A, B1, S12, B2, batch_size=1000):
            n = A.shape[0]
            total = 0.0

            if use_torch:
                B1S12 = B1 @ S12  # (n, r)
            else:
                B1S12 = B1 @ S12

            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)

                if use_torch:
                    A_batch = A[i:end]
                    recon = B1S12[i:end] @ B2.T
                    total += l21_norm_torch(A_batch - recon)
                else:
                    if sp.issparse(A):
                        A_batch = A[i:end].toarray()
                    else:
                        A_batch = A[i:end]
                    recon = B1S12[i:end] @ B2.T
                    total += l21_norm_np(A_batch - recon)

            return total

        # ============= 3. Frobenius Diff ============= #
        def efficient_frobenius_diff(U, B):
            if use_torch:
                UTU = U.T @ U
                BTB = B.T @ B
                UTB = U.T @ B

                term1 = torch.norm(UTU, p="fro") ** 2
                term2 = 2 * torch.norm(UTB, p="fro") ** 2
                term3 = torch.norm(BTB, p="fro") ** 2
                return term1 - term2 + term3
            else:
                UTU = U.T @ U
                BTB = B.T @ B
                UTB = U.T @ B
                return (
                    np.linalg.norm(UTU, "fro") ** 2
                    - 2 * np.linalg.norm(UTB, "fro") ** 2
                    + np.linalg.norm(BTB, "fro") ** 2
                )

        # ============= 计算各项损失 ============= #
        attributes_layer_loss = efficient_l21_norm_intra(self.la, self.U1)
        structure_layer_loss = efficient_l21_norm_intra(self.ls, self.U2)
        intra_loss = attributes_layer_loss + structure_layer_loss

        inter_loss = self.mu1 * efficient_l21_norm_inter(
            self.li, self.B1, self.S12, self.B2
        )

        sim_loss = self.mu2 * (
            efficient_frobenius_diff(self.U1, self.B1)
            + efficient_frobenius_diff(self.U2, self.B2)
        )

        total_loss = intra_loss + inter_loss + sim_loss

        # 注意：Torch tensor 转 python float
        if use_torch:
            total_loss_value = total_loss.item()
            att_v = attributes_layer_loss.item()
            str_v = structure_layer_loss.item()
            intra_v = intra_loss.item()
            inter_v = inter_loss.item()
            sim_v = sim_loss.item()
        else:
            total_loss_value = total_loss
            att_v = attributes_layer_loss
            str_v = structure_layer_loss
            intra_v = intra_loss
            inter_v = inter_loss
            sim_v = sim_loss

        # ====== 记录损失 ====== #
        self.loss_tracker.step(
            total_loss_value,
            att_v,
            str_v,
            intra_v,
            inter_v,
            sim_v,
        )

        return total_loss_value

    def build_Z(self, A, U, eps, batch_size=512):
        """构建对角矩阵 Z = diag(1 / ||A - UUᵀ||₂)，用于加权重构误差

        参数:
            A: 原始邻接矩阵
            U: 潜在特征矩阵

        返回值:
            numpy.ndarray: 对角权重矩阵
        """
        # # 计算残差矩阵（原始矩阵与重构矩阵的差）
        # residual = A - U @ U.T
        # # 计算每行的L2范数，并添加小值防止除零
        # norms = np.linalg.norm(residual, axis=1) + self.eps
        # # 构建对角矩阵，对角线元素为范数的倒数
        # return np.diag(1 / norms)
        n = A.shape[0]
        z = np.empty(n, dtype=float)
        # 如果 A 是稀疏，把它转为密集的行访问形式（但我们仍只取需要的行）
        if sp.issparse(A):
            A_dense = None  # we will use .toarray() per-batch to avoid huge allocation
            get_row = lambda idx: (A[idx].toarray())
        else:
            A_dense = A
            get_row = lambda idx: A_dense[idx]

        # Precompute U (n x r)
        # 对于 batch 中索引 idx_batch (长度 b)，计算 reconstruction rows:
        # recon_batch = (U @ U[idx_batch].T).T  -> shape (b, n)
        # 这样我们不需要生成 n x n 全矩阵，而是 n x b 中间量
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            idx = np.arange(start, end)
            U_batch = U[idx, :]  # (b x r)
            # recon = U @ U_batch.T  -> (n x b); 我们需要每个 batch 行对应的向量 recon[:, j]
            recon = U @ U_batch.T  # (n x b)
            # 转置得到 (b x n), 与 A 的这些行一一对应
            recon_rows = recon.T  # (b x n)
            A_rows = get_row(idx)  # (b x n)
            # residual rows
            residual = A_rows - recon_rows  # (b x n)
            norms = np.linalg.norm(residual, axis=1) + eps  # length b
            z[idx] = 1.0 / norms
        return z  # 长度 n 的向量

    def build_Z12(self, A12, B1, B2, S12, eps, batch_size=512):
        """为层间重构构建对角矩阵

        参数:
            A12: 层间连接的邻接矩阵
            B1: 第一层网络的共享嵌入矩阵
            B2: 第二层网络的共享嵌入矩阵
            S12: 层间映射矩阵

        返回值:
            numpy.ndarray: 对角权重矩阵
        """
        # # 计算层间残差矩阵
        # residual = A12 - B1 @ S12 @ B2.T
        # # 计算每行的L2范数，并添加小值防止除零
        # norms = np.linalg.norm(residual, axis=1) + eps
        # # 构建对角矩阵，对角线元素为范数的倒数
        # return np.diag(1 / norms)
        n = A12.shape[0]
        z = np.empty(n, dtype=float)
        if sp.issparse(A12):
            get_row = lambda idx: A12[idx].toarray()
        else:
            A12_dense = A12
            get_row = lambda idx: A12_dense[idx]

        # precompute small matrices if possible
        # We'll compute reconstruction rows similarly by batches:
        # for idx_batch, recon_rows = (B1 @ S12 @ B2.T)[idx_batch, :] = ( (B1 @ S12) @ B2.T )[idx_batch,:]
        # For batch, do: temp = (B1 @ S12)  (n1 x r) ; recon = temp @ B2.T -> (n1 x n2) but we compute per batch
        temp = B1 @ S12  # (n1 x r)
        # now batch over rows of temp
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            idx = np.arange(start, end)
            temp_batch = temp[idx, :]  # (b x r)
            recon = temp_batch @ B2.T  # (b x n)  (b rows of reconstruction)
            A_rows = get_row(idx)  # (b x n)
            residual = A_rows - recon
            norms = np.linalg.norm(residual, axis=1) + eps
            z[idx] = 1.0 / norms
        return z  # 长度 n 的向量

    def updateU1_numpy(self, z1, z2, z12, eps):
        # 分子部分：包含重构误差和与B1的一致性约束
        U1_num = (
            (z1[:, np.newaxis] * self.la) @ self.U1
            + self.la @ (z1[:, np.newaxis] * self.U1)
            + 2 * self.mu2 * self.B1 @ (self.B1.T @ self.U1)
        ) * self.U1

        # 分母部分：包含归一化项
        U1_den = (
            self.U1 @ (self.U1.T @ (z1[:, np.newaxis] * self.U1))
            + z1[:, np.newaxis] * (self.U1 @ (self.U1.T @ self.U1))
            + 2 * self.mu2 * self.U1 @ (self.U1.T @ self.U1)
            + eps
        )

        # 执行更新
        return U1_num / U1_den

    def updateU1_torch(self, z1, z2, z12, eps=1e-10):
        """
        使用 PyTorch (GPU 可加速) 的 U1 更新。
        仅基于原始 updateU1 的公式结构，无数学修改。
        """

        # ------- 保证张量在同一 device -------
        device = self.U1.device
        # z1 是 numpy? -> 转为 torch
        if not isinstance(z1, torch.Tensor):
            z1 = torch.tensor(z1, dtype=self.U1.dtype, device=device)

        z1_col = z1[:, None]  # (n, 1)

        # --- 分子 numerator ---
        # (z1[:, np.newaxis] * self.la) @ self.U1
        term1 = (z1_col * self.la) @ self.U1

        # self.la @ (z1[:, np.newaxis] * self.U1)
        term2 = self.la @ (z1_col * self.U1)

        # 2 * mu2 * B1 @ (B1.T @ U1)
        term3 = 2 * self.mu2 * self.B1 @ (self.B1.T @ self.U1)

        U1_num = (term1 + term2 + term3) * self.U1

        # --- 分母 denominator ---
        # U1 @ (U1.T @ (z1[:, np.newaxis] * U1))
        den1 = self.U1 @ (self.U1.T @ (z1_col * self.U1))

        # z1[:, np.newaxis] * (U1 @ (U1.T @ U1))
        den2 = z1_col * (self.U1 @ (self.U1.T @ self.U1))

        # 2 * mu2 * U1 @ (U1.T @ U1)
        den3 = 2 * self.mu2 * self.U1 @ (self.U1.T @ self.U1)

        U1_den = den1 + den2 + den3 + eps

        return U1_num / U1_den

    def updateU1(self, z1, z2, z12, eps=1e-10):
        if self.use_gpu:
            return self.updateU1_torch(z1, z2, z12, eps).detach().cpu().numpy()
        else:
            return self.updateU1_numpy(z1, z2, z12, eps)

    def update_U1_old(self, Z1, Z2, Z12, eps=1e-10):
        """原始版本的U1更新方法"""
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
        return U1_num / U1_den

    def updateU2_numpy(self, z1, z2, z12, eps):
        # 分子部分：包含重构误差和与B2的一致性约束
        U2_num = (
            (z2[:, np.newaxis] * self.ls) @ self.U2
            + self.ls @ (z2[:, np.newaxis] * self.U2)
            + 2 * self.mu2 * self.B2 @ (self.B2.T @ self.U2)
        ) * self.U2

        # 分母部分：包含归一化项
        U2_den = (
            self.U2 @ (self.U2.T @ (z2[:, np.newaxis] * self.U2))
            + z2[:, np.newaxis] * (self.U2 @ (self.U2.T @ self.U2))
            + 2 * self.mu2 * self.U2 @ (self.U2.T @ self.U2)
            + eps
        )

        # 执行更新
        return U2_num / U2_den

    def updateU2_torch(self, z1, z2, z12, eps=1e-10):
        """
        使用 PyTorch (GPU 可加速) 的 U2 更新。
        完全等价于原 numpy 版本，不改变数学更新规则。
        """

        device = self.U2.device

        # --- z2 转 torch ---
        if not isinstance(z2, torch.Tensor):
            z2 = torch.tensor(z2, dtype=self.U2.dtype, device=device)

        z2_col = z2[:, None]  # (n, 1)

        # ==================================================
        #                  Numerator 分子
        # ==================================================

        # (z2 * ls) @ U2
        term1 = (z2_col * self.ls) @ self.U2

        # ls @ (z2 * U2)
        term2 = self.ls @ (z2_col * self.U2)

        # 2 * mu2 * B2 @ (B2.T @ U2)
        term3 = 2 * self.mu2 * self.B2 @ (self.B2.T @ self.U2)

        U2_num = (term1 + term2 + term3) * self.U2

        # ==================================================
        #                  Denominator 分母
        # ==================================================

        # U2 @ (U2.T @ (z2 * U2))
        den1 = self.U2 @ (self.U2.T @ (z2_col * self.U2))

        # z2 * (U2 @ (U2.T @ U2))
        den2 = z2_col * (self.U2 @ (self.U2.T @ self.U2))

        # 2 * mu2 * U2 @ (U2.T @ U2)
        den3 = 2 * self.mu2 * self.U2 @ (self.U2.T @ self.U2)

        U2_den = den1 + den2 + den3 + eps

        return U2_num / U2_den

    def updateU2(self, z1, z2, z12, eps=1e-10):
        if self.use_gpu:
            return self.updateU2_torch(z1, z2, z12, eps)
        else:
            return self.updateU2_numpy(z1, z2, z12, eps)

    def update_U2_old(self, Z1, Z2, Z12, eps=1e-10):
        """原始版本的U2更新方法"""
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

        return U2_num / U2_den

    def updateB1_numpy(self, z1, z2, z12, eps):
        B2_ST = self.B2 @ self.S12.T  # n * r
        B1_num = (
            self.mu1 * self.li @ (z12[:, np.newaxis] * B2_ST)
            + 2 * self.mu2 * self.U1 @ (self.U1.T @ self.B1)
        ) * self.B1
        B1_den = (
            self.mu1 * self.B1 @ (B2_ST.T) @ (z12[:, np.newaxis] * B2_ST)
            + 2 * self.mu2 * self.B1 @ (self.B1.T @ self.B1)
            + eps
        )
        return B1_num / B1_den

    def updateB1_torch(self, z1, z2, z12, eps=1e-10):
        """
        GPU 加速版本的 B1 更新 (PyTorch)
        数学公式完全和原 numpy 版本一致。
        """

        device = self.B1.device

        # ---- z12 转换为 torch ----
        if not isinstance(z12, torch.Tensor):
            z12 = torch.tensor(z12, dtype=self.B1.dtype, device=device)

        z12_col = z12[:, None]  # (n, 1)

        # ==========================================================
        #                      Numerator 分子
        # ==========================================================

        # B2_ST = B2 @ S12.T   (n × r)
        B2_ST = self.B2 @ self.S12.T

        # μ1 * li @ (z12 * B2_ST)
        term1 = self.mu1 * self.li @ (z12_col * B2_ST)

        # 2 * μ2 * U1 @ (U1.T @ B1)
        term2 = 2 * self.mu2 * self.U1 @ (self.U1.T @ self.B1)

        B1_num = (term1 + term2) * self.B1  # Hadamard product

        # ==========================================================
        #                      Denominator 分母
        # ==========================================================

        # B1 @ (B2_ST.T @ (z12 * B2_ST))
        den1 = self.mu1 * self.B1 @ (B2_ST.T @ (z12_col * B2_ST))

        # 2 * μ2 * B1 @ (B1.T @ B1)
        den2 = 2 * self.mu2 * self.B1 @ (self.B1.T @ self.B1)

        B1_den = den1 + den2 + eps

        return B1_num / B1_den

    def updateB1(self, z1, z2, z12, eps=1e-10):
        if self.use_gpu:
            return self.updateB1_torch(z1, z2, z12, eps)
        else:
            return self.updateB1_numpy(z1, z2, z12, eps)

    def update_B1_old(self, Z1, Z2, Z12, eps=1e-10):
        """原始版本的B1更新方法"""
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

        return B1_num / B1_den

    def updateB2_numpy(self, z1, z2, z12, eps):
        B1_S = self.B1 @ self.S12
        # 更新B2矩阵：与B1类似的更新规则
        B2_num = (
            self.mu1 * z12[:, np.newaxis] * (self.li.T @ B1_S)
            + 2 * self.mu2 * self.U2 @ (self.U2.T @ self.B2)
        ) * self.B2
        B2_den = (
            self.mu1 * z12[:, np.newaxis] * (self.B2 @ ((B1_S.T) @ B1_S))
            + 2 * self.mu2 * self.B2 @ (self.B2.T @ self.B2)
            + eps
        )
        return B2_num / B2_den

    def updateB2_torch(self, z1, z2, z12, eps=1e-10):
        """
        GPU 加速版本的 B2 更新 (PyTorch)
        数学公式完全和原 numpy 版本一致。
        """

        device = self.B2.device

        # ---- z12 转换为 torch ----
        if not isinstance(z12, torch.Tensor):
            z12 = torch.tensor(z12, dtype=self.B2.dtype, device=device)

        z12_col = z12[:, None]  # (n, 1)

        # ==========================================================
        #                     Numerator 分子
        # ==========================================================

        # B1_S = B1 @ S12     (n × r)
        B1_S = self.B1 @ self.S12

        # μ1 * z12 * (li.T @ B1_S)
        term1 = self.mu1 * z12_col * (self.li.T @ B1_S)

        # 2 * μ2 * U2 @ (U2.T @ B2)
        term2 = 2 * self.mu2 * self.U2 @ (self.U2.T @ self.B2)

        B2_num = (term1 + term2) * self.B2  # Hadamard product

        # ==========================================================
        #                     Denominator 分母
        # ==========================================================

        # B2 @ (B1_S.T @ B1_S)
        term_d1 = self.B2 @ (B1_S.T @ B1_S)

        # μ1 * z12 * term_d1
        den1 = self.mu1 * z12_col * term_d1

        # 2 * μ2 * B2 @ (B2.T @ B2)
        den2 = 2 * self.mu2 * self.B2 @ (self.B2.T @ self.B2)

        B2_den = den1 + den2 + eps

        return B2_num / B2_den

    def updateB2(self, z1, z2, z12, eps=1e-10):
        if self.use_gpu:
            return self.updateB2_torch(z1, z2, z12, eps)
        else:
            return self.updateB2_numpy(z1, z2, z12, eps)

    def update_B2_old(self, Z1, Z2, Z12, eps=1e-10):
        """原始版本的B2更新方法"""
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

        return B2_num / B2_den

    def updateS_numpy(self, z1, z2, z12, eps):
        # 更新S12矩阵：层间映射矩阵的更新规则
        S12_num = self.B1.T @ (self.li @ (z12[:, np.newaxis] * self.B2))
        S12_den = (
            self.B1.T @ self.B1 @ self.S12 @ self.B2.T @ (z12[:, np.newaxis] * self.B2)
            + eps
        )
        return (S12_num / S12_den) * self.S12

    def updateS_torch(self, z1, z2, z12, eps=1e-10):
        """
        GPU 加速版本的 S12 更新 (PyTorch)
        数学结构完全与原 numpy 版本相同。
        """

        device = self.S12.device

        # ---- z12 转 torch 张量 ----
        if not isinstance(z12, torch.Tensor):
            z12 = torch.tensor(z12, dtype=self.S12.dtype, device=device)

        z12_col = z12[:, None]  # (n, 1)

        # ==========================================================
        #                         Numerator
        # ==========================================================

        # B1.T @ (li @ (z12 * B2))
        S12_num = self.B1.T @ (self.li @ (z12_col * self.B2))

        # ==========================================================
        #                         Denominator
        # ==========================================================

        # B1.T @ B1 @ S12 @ B2.T @ (z12 * B2)
        S12_den = (
            self.B1.T @ self.B1 @ self.S12 @ (self.B2.T @ (z12_col * self.B2)) + eps
        )

        # element-wise / and element-wise *
        return (S12_num / S12_den) * self.S12

    def updateS(self, z1, z2, z12, eps=1e-10):
        if self.use_gpu:
            return self.updateS_torch(z1, z2, z12, eps)
        else:
            return self.updateS_numpy(z1, z2, z12, eps)

    def update_S12_old(self, Z1, Z2, Z12, eps=1e-10):
        """原始版本的S12更新方法"""
        # 更新S12矩阵：层间映射矩阵的更新规则
        S12_num = self.B1.T @ self.li @ Z12 @ self.B2
        S12_den = self.B1.T @ self.B1 @ self.S12 @ self.B2.T @ Z12 @ self.B2 + eps
        return (S12_num / S12_den) * self.S12

    def multipicateUpdate_old(self, eps=1e-10):
        """原始版本的完整更新方法"""

        # 构建Z矩阵
        z1 = np.diag(self.build_Z(self.la, self.U1, eps))
        z2 = np.diag(self.build_Z(self.ls, self.U2, eps))
        z12 = np.diag(self.build_Z12(self.li, self.B1, self.B2, self.S12, eps))

        # 分别更新每个矩阵
        self.U1 = self.update_U1_old(z1, z2, z12, eps)
        self.U2 = self.update_U2_old(z1, z2, z12, eps)
        self.B1 = self.update_B1_old(z1, z2, z12, eps)
        self.B2 = self.update_B2_old(z1, z2, z12, eps)
        self.S12 = self.update_S12_old(z1, z2, z12, eps)

        return self.U1, self.U2, self.B1, self.B2, self.S12

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

        # 构建三层的对角权重矩阵, 但是算法通过向量广播改进。
        z1, z2, z12 = (
            self.build_Z(self.la, self.U1, eps),
            self.build_Z(self.ls, self.U2, eps),
            self.build_Z12(self.li, self.B1, self.B2, self.S12, eps),
        )

        self.U1 = self.updateU1(z1, z2, z12, eps)
        self.U2 = self.updateU2(z1, z2, z12, eps)
        self.B1 = self.updateB1(z1, z2, z12, eps)
        self.B2 = self.updateB2(z1, z2, z12, eps)
        self.S12 = self.updateS(z1, z2, z12, eps)

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
                if (it + 1) % 10 == 0:
                    console.print(
                        f"[Update] iteration={it+1}, loss={loss:.4f}, best_loss={best_loss:.4f}, computing_time={time.time() - time_start:.4f} s/it",
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

            # --- Torch 版本 ---
            if isinstance(Z, torch.Tensor):
                return torch.argmax(Z, dim=1).detach().cpu().numpy()

            # --- NumPy 版本 ---
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
