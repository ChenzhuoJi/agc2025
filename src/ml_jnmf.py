import numpy as np
import pandas as pd
from sklearn.decomposition._nmf import _initialize_nmf
from sklearn.cluster import KMeans
from rich.console import Console

from src.helpers import compute_communitude_metric


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


class ML_JNMF:
    """
    Core implementation of Multi-Level Joint Non-negative Matrix Factorization (ML-JNMF).
    The algorithm jointly factorizes intra-layer and inter-layer adjacency matrices
    to learn shared latent embeddings across multiple networks.
    """

    def __init__(
        self,
        mu1=1.0,
        mu2=2.0,
        max_iter=300,
        tol=1e-4,
        patience=20,
        min_delta=1e-4,
        random_state=42,
    ):
        """
        Args:
            mu1 (float): Weight for cross-layer reconstruction constraint.
            mu2 (float): Weight for intra/inter-layer embedding similarity constraint.
            max_iter (int): Maximum number of update iterations.
            tol (float): Relative tolerance for convergence check.
            patience (int): Number of iterations to wait for improvement before early stopping.
            min_delta (float): Minimum relative improvement in loss required to stop.
        """
        self.mu1 = mu1
        self.mu2 = mu2
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)
        self.convergence_checker = ConvergenceChecker(
            patience=patience,
        )
        # 模型的结果
        self.U1, self.U2, self.B1, self.B2, self.S12 = [None] * 5
        self.loss_history = []
        self.final_loss = None
        self.is_early_stopped = False


    def _initialize(self, A1, A2, A12, r):
        """Initialize U1, U2, B1, B2, S12 using NNDSVDAR for stability."""
        U1, _ = _initialize_nmf(
            A1, n_components=r, init="nndsvdar", random_state=self.random_state
        )
        U2, _ = _initialize_nmf(
            A2, n_components=r, init="nndsvdar", random_state=self.random_state
        )
        B1, B2t = _initialize_nmf(
            A12, n_components=r, init="nndsvdar", random_state=self.random_state
        )
        B2 = B2t.T
        # 处理 NaN 值
        U1 = np.nan_to_num(U1, nan=1e-6)
        U2 = np.nan_to_num(U2, nan=1e-6)
        B1 = np.nan_to_num(B1, nan=1e-6)
        B2 = np.nan_to_num(B2, nan=1e-6)

        # ceil = 2
        # U1 = np.random.uniform(0, ceil, size=(A1.shape[0], r))
        # U2 = np.random.uniform(0, ceil, size=(A2.shape[0], r))
        # B1 = np.random.uniform(0, ceil, size=(A1.shape[1], r))
        # B2 = np.random.uniform(0, ceil, size=(A2.shape[1], r))
        S12 = np.eye(r)
        return U1, U2, B1, B2, S12

    def _compute_loss(self, A1, A2, A12, U1, U2, B1, B2, S12):
        """Compute the total objective function of ML-JNMF."""

        # 计算 L2,1 范数的辅助函数
        def l21_norm(X):
            return np.sum(np.sqrt(np.sum(X**2, axis=1)))

        # -------- Intra loss (use L2,1 norm, no square) --------
        intra_loss = l21_norm(A1 - U1 @ U1.T) + l21_norm(A2 - U2 @ U2.T)

        # -------- Inter loss (use L2,1 norm, no square) --------
        inter_loss = self.mu1 * l21_norm(A12 - B1 @ S12 @ B2.T)

        # -------- Sim loss (still Frobenius norm squared) --------
        sim_loss = self.mu2 * (
            np.linalg.norm(U1 @ U1.T - B1 @ B1.T, "fro") ** 2
            + np.linalg.norm(U2 @ U2.T - B2 @ B2.T, "fro") ** 2
        )

        total_loss = intra_loss + inter_loss + sim_loss
        if np.isnan(total_loss) or np.isinf(total_loss):
            print("⚠️ Loss NaN detected")
            print("A1:", np.isnan(A1).any(), "A2:", np.isnan(A2).any(), "A12:", np.isnan(A12).any())
            print("U1:", np.isnan(U1).any(), "U2:", np.isnan(U2).any())
            print("B1:", np.isnan(B1).any(), "B2:", np.isnan(B2).any(), "S12:", np.isnan(S12).any())
            print("Any Inf:", np.isinf(U1).any() or np.isinf(U2).any() or np.isinf(B1).any())
            raise ValueError("NaN detected in loss computation")
        return total_loss

    def _update(self, A1, A2, A12, U1, U2, B1, B2, S12, eps=1e-10):
        """执行一次所有因子的乘法更新迭代

        此方法是ML-JNMF算法的核心更新步骤，实现了非负矩阵分解中的乘法更新规则，
        用于迭代优化模型参数。

        参数:
            A1: 第一层网络的邻接矩阵
            A2: 第二层网络的邻接矩阵
            A12: 层间连接的邻接矩阵
            U1: 第一层网络的潜在特征矩阵（待更新）
            U2: 第二层网络的潜在特征矩阵（待更新）
            B1: 第一层网络的共享嵌入矩阵（待更新）
            B2: 第二层网络的共享嵌入矩阵（待更新）
            S12: 层间映射矩阵（待更新）
            eps: 防止除零错误的小值，默认为1e-10

        返回值:
            tuple: 包含更新后的矩阵 (U1, U2, B1, B2, S12)
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
        Z1, Z2, Z12 = build_Z(A1, U1), build_Z(A2, U2), build_Z12(A12, B1, B2, S12)

        # 更新U1矩阵：使用乘法更新规则，确保非负性
        # 分子部分：包含重构误差和与B1的一致性约束
        U1_num = (Z1 @ A1 @ U1 + A1 @ Z1 @ U1 + 2 * self.mu2 * B1 @ B1.T @ U1) * U1
        # 分母部分：包含归一化项
        U1_den = (
            U1 @ U1.T @ Z1 @ U1
            + Z1 @ U1 @ U1.T @ U1
            + 2 * self.mu2 * U1 @ U1.T @ U1
            + eps
        )
        # 执行更新
        U1 = U1_num / U1_den

        # 更新U2矩阵：与U1类似的更新规则
        U2_num = (Z2 @ A2 @ U2 + A2 @ Z2 @ U2 + 2 * self.mu2 * B2 @ B2.T @ U2) * U2
        U2_den = (
            U2 @ U2.T @ Z2 @ U2
            + Z2 @ U2 @ U2.T @ U2
            + 2 * self.mu2 * U2 @ U2.T @ U2
            + eps
        )
        U2 = U2_num / U2_den

        # 更新B1矩阵：结合层间重构和与U1的一致性约束
        B1_num = (
            self.mu1 * A12 @ Z12 @ B2 @ S12.T + 2 * self.mu2 * U1 @ U1.T @ B1
        ) * B1
        B1_den = (
            self.mu1 * B1 @ S12 @ B2.T @ Z12 @ B2 @ S12.T
            + 2 * self.mu2 * B1 @ B1.T @ B1
            + eps
        )
        B1 = B1_num / B1_den

        # 更新B2矩阵：与B1类似的更新规则
        B2_num = (
            self.mu1 * Z12 @ A12.T @ B1 @ S12 + 2 * self.mu2 * U2 @ U2.T @ B2
        ) * B2
        B2_den = (
            self.mu1 * Z12 @ B2 @ S12.T @ B1.T @ B1 @ S12
            + 2 * self.mu2 * B2 @ B2.T @ B2
            + eps
        )
        B2 = B2_num / B2_den

        # 更新S12矩阵：层间映射矩阵的更新规则
        S12_num = B1.T @ A12 @ Z12 @ B2
        S12_den = B1.T @ B1 @ S12 @ B2.T @ Z12 @ B2 + eps
        S12 = (S12_num / S12_den) * S12

        # 返回所有更新后的矩阵
        return U1, U2, B1, B2, S12

    def fit(self, A1, A2, A12, r):
        """
        Fit ML-JNMF on given adjacency matrices using EarlyStopping and ConvergenceChecker.
        """
        console = Console()
        # Initialize matrices
        self.A1, self.A2, self.A12 = A1, A2, A12
        self.U1, self.U2, self.B1, self.B2, self.S12 = self._initialize(A1, A2, A12, r)

        # 初始化训练管理器
        best_loss = float("inf")
        best_params = None
        self.early_stopper.reset()
        self.convergence_checker.reset()

        for it in range(self.max_iter):
            # 计算当前 loss
            loss = self._compute_loss(
                self.A1, self.A2, self.A12, self.U1, self.U2, self.B1, self.B2, self.S12
            )
            self.loss_history.append(loss)
            # console.print(
            #     f"iteration={it+1}, loss={loss:.4f}, best_loss={best_loss:.4f}",
            #     style="bold blue",
            # )
            # 🔍 Early stopping 检查放在更新 best_loss 之前
            if self.early_stopper.step(loss, best_loss):
                # console.print(f"loss={loss:.4f}, best_loss={best_loss:.4f}")
                self.U1, self.U2, self.B1, self.B2, self.S12 = best_params
                self.is_early_stopped = True
                self.final_loss = best_loss
                console.print(
                    f"[Early Stop] iteration={it+1}, best_loss={best_loss:.4f} at iteration {best_it+1}, n_nodes={A1.shape[0]}",
                    style="bold yellow",
                )
                break

            # ✅ 在 step() 之后再更新 best_loss
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
                console.print(
                    f"[Converged] iteration={it+1}, loss={loss:.4f}",
                    style="bold green",
                )
                break

            # 更新因子矩阵
            self.U1, self.U2, self.B1, self.B2, self.S12 = self._update(
                A1, A2, A12, self.U1, self.U2, self.B1, self.B2, self.S12
            )

        # 如果循环自然结束，也使用最佳参数
        else:
            self.U1, self.U2, self.B1, self.B2, self.S12 = best_params
            self.final_loss = best_loss
            console.print(
                f"[End] iteration={it+1}, loss={loss:.4f}",
                style="bold red",
            )
        return self

    def predict(self, r, pred_method, lamb=None):
        """
        基于训练好的模型参数预测社区结构
        
        参数:
        r: int - 社区数量/聚类数目
        pred_method: str - 预测方法，可选值包括 'kmeans' 和 'laplace'
        lamb: float (可选) - 用于加权组合U1和U2的参数，范围在[0,1]之间
        """
        if pred_method == "kmeans":
            kmeans = KMeans(n_clusters=r, random_state=self.random_state)
            Z = lamb * self.U1 + (1 - lamb) * self.U2
            S = np.dot(Z, Z.T)
            return kmeans.fit_predict(Z)

        elif pred_method == "laplace":
            Z = lamb * self.U1 + (1 - lamb) * self.U2
            S = np.dot(Z, Z.T)
            D = np.sum(S, axis=1)
            diag_inv_sqrt = np.where(D > 1e-10, 1.0 / np.sqrt(D), 0.0)
            D_inv_sqrt = np.diag(diag_inv_sqrt)
            L = np.eye(S.shape[0]) - D_inv_sqrt.dot(S).dot(D_inv_sqrt)
            _, eigenvectors = np.linalg.eigh(L)

            Y = eigenvectors[:, :r]  # 每一列是一个特征向量
            kmeans = KMeans(n_clusters=r, random_state=self.random_state)
            return kmeans.fit_predict(Y)

        # -------- Obtain intra-layer and inter-layer community labels --------
        label_intra_1 = np.argmax(self.U1, axis=1)
        label_inter_1 = np.argmax(self.B1, axis=1)
        label_intra_2 = np.argmax(self.U2, axis=1)
        label_inter_2 = np.argmax(self.B2, axis=1)

        # -------- Calculate community metrics --------
        comm_intra_1 = compute_communitude_metric(self.A1, label_intra_1)
        comm_intra_2 = compute_communitude_metric(self.A2, label_intra_2)
        comm_inter_1 = compute_communitude_metric(self.A12, label_inter_1, axis=0)
        comm_inter_2 = compute_communitude_metric(self.A12, label_inter_2, axis=1)

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
        return df1, df2

    def fit_predict(
        self,
        A1,
        A2,
        A12,
        r,
        pred_method,
        lamb=None,
    ):
        self.fit(A1, A2, A12, r)
        cluster_labels = self.predict(r, pred_method, lamb)
        return cluster_labels
