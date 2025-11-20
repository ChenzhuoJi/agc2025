import argparse
import datetime
import json
from typing import Literal, Dict, Any
import time
import zipfile
import os
import traceback

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table
from rich import box

from src.ml_jnmf import ML_JNMF
from src.processor import feature_process, high_order, featjson2sparse
from src.helpers import determine_community_number

preprocessParameters = {"feature_kernel": "rbf", "order": 5, "decay": 2}

mfParameters = {"interWeight": 4, "pairwiseWeight": 2}

pred_method = "lambda"

defalut_dataname = "cora"
array_dtype = np.float32

console = Console()


def build_folders():
    os.makedirs("results/clustering/", exist_ok=True)
    os.makedirs("results/", exist_ok=True)
    os.makedirs("results/logs/", exist_ok=True)
    os.makedirs("data/", exist_ok=True)
    os.makedirs("precomputed/", exist_ok=True)
    os.makedirs("results/best_r/", exist_ok=True)


def extract_data():
    zip_file = "stgraphs.zip"
    out_dir = "stgraphs"

    # 如果输出目录不存在则创建
    os.makedirs(out_dir, exist_ok=True)

    # 开始解压
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(out_dir)

    print(f"解压完成 → {out_dir}/")


def load_data(dataname: str):
    console = Console()

    # 加载数据
    feature_file = f"stgraphs/{dataname}.features"
    features_sparse = featjson2sparse(feature_file)

    # 读取边（假设格式：u,v 每行一个边）
    edges_file = f"stgraphs/{dataname}.edges"
    edges = np.loadtxt(edges_file, delimiter=",", dtype=int)

    # 节点总数（假设 ID 基于 0）
    n = edges.max() + 1

    # 构建双向边
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.ones(len(row), dtype=np.float32)

    # 构建稀疏邻接矩阵
    adj_matrix_sparse = sp.csr_matrix(
        (data, (row, col)), shape=(n, n), dtype=array_dtype
    )
    # adj_matrix_sparse.setdiag(1.0)
    is_symmetric = (adj_matrix_sparse != adj_matrix_sparse.T).nnz == 0
    if not is_symmetric:
        console.print(f"[bold red]警告: 图 {dataname} 不是对称的![/bold red]")

    # 创建简约表格
    table = Table(show_header=True, header_style="default", box=box.SIMPLE)
    table.add_column("矩阵")
    table.add_column("形状")
    table.add_column("数据格式")
    table.add_column("稀疏度")
    table.add_column("内存")

    # 计算内存占用（近似）
    features_memory = (
        features_sparse.data.nbytes
        + features_sparse.indices.nbytes
        + features_sparse.indptr.nbytes
    ) / 1024
    adj_memory = (
        adj_matrix_sparse.data.nbytes
        + adj_matrix_sparse.indices.nbytes
        + adj_matrix_sparse.indptr.nbytes
    ) / 1024

    table.add_row(
        "特征矩阵",
        f"{features_sparse.shape[0]}×{features_sparse.shape[1]}",
        f"{type(features_sparse).__name__} {features_sparse.dtype}",
        f"{features_sparse.nnz/(features_sparse.shape[0]*features_sparse.shape[1]):.4%}",
        f"{features_memory:.1f} KB",
    )

    table.add_row(
        "邻接矩阵",
        f"{adj_matrix_sparse.shape[0]}×{adj_matrix_sparse.shape[1]}",
        f"{type(adj_matrix_sparse).__name__} {adj_matrix_sparse.dtype}",
        f"{adj_matrix_sparse.nnz/(adj_matrix_sparse.shape[0]*adj_matrix_sparse.shape[1]):.4%}",
        f"{adj_memory:.1f} KB",
    )

    # 显示信息
    console.print(f"\n数据集: {dataname}")
    console.print(f"节点数量: {adj_matrix_sparse.shape[0]}")
    console.print(f"边数量: {edges.shape[0]}")
    console.print(table)

    return features_sparse, adj_matrix_sparse


# TODO: 把论文中的calculate_r实现过来
def calculate_r(la, ls, li, dataname):
    # 自动确定最优社区数量
    if os.path.exists(f"results/best_r/{dataname}.txt"):
        with open(f"results/best_r/{dataname}.txt", "r") as f:
            best_r = int(f.read().strip())
    else:
        best_r = determine_community_number(
            li, max_r=10, save_path=f"results/best_r/{dataname}",
        )
        with open(f"results/best_r/{dataname}.txt", "w") as f:
            f.write(f"{best_r}")
    return best_r


# TODO: 优化preprocess，主要优化联合矩阵的计算时间
# 方案一：预处理后，全都转化为csr_matrix，非常慢 ❌
# 方案二：预处理后，全都转化为np.ndarray ✅，citeseer:2.1816s, 但是大矩阵的高阶运算时间还是会很大。最大占用内存约20GB，在处理30000节点的数据时出现
# 进一步优化，发现ls稀疏性并不是都很好，内存占用有时候比稠密矩阵要大，全都转换为np.ndarray
# 改进成了Preprocessor类，调用run方法进行预处理
class Preprocessor:
    """
    一个专门用于构建三层网络（属性层、结构层、联合层）的预处理类。
    """

    def __init__(
        self,
        features_sparse: sp.csr_matrix,
        adj_matrix_sparse: sp.csr_matrix,
        preprocess_params: dict,
        edge_undirected: bool = True,
        dataname: str = None,
    ):
        """
        初始化预处理实例。

        Args:
            features_sparse (sp.csr_matrix): 稀疏特征矩阵。
            adj_matrix_sparse (sp.csr_matrix): 稀疏邻接矩阵。
            preprocess_params (dict): 预处理参数。
            edge_undirected (bool, optional): 图是否为无向图。默认 True。
            dataname (str, optional): 数据集名称，用于缓存文件命名。默认 None。
        """
        self.features_sparse = features_sparse
        self.adj_matrix_sparse = adj_matrix_sparse
        self.preprocess_params = preprocess_params
        self.edge_undirected = edge_undirected
        self.dataname = dataname

        # 从参数中提取配置
        self.kernel = self.preprocess_params.get("feature_kernel")
        self.order = self.preprocess_params.get("order")
        self.decay = self.preprocess_params.get("decay")

        # 初始化状态
        self.la = None
        self.ls = None
        self.li = None

        self.la_time = 0.0
        self.ls_time = 0.0
        self.li_time = 0.0

        # 缓存相关
        self.use_cache = self.dataname is not None
        self.cache_dir = "precomputed"
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def run(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        执行完整的预处理流程，构建并返回三层矩阵。
        """
        print("Starting preprocessing...")

        self._build_similarity_matrix()
        self._build_high_order_matrix()
        self._build_joint_matrix()

        self._print_summary_table()

        print("Preprocessing completed.")
        return self.la, self.ls, self.li

    def _build_similarity_matrix(self):
        """构建属性相似度矩阵 (la)。"""
        print("  Building similarity matrix (la)...")

        def compute_la():
            X = self.features_sparse.tocsr()
            la = feature_process(X, self.kernel)
            la = la.astype(array_dtype)
            max_val = np.diag(la).max()
            if max_val > 0:
                la /= max_val
            return la

        filepath = (
            f"{self.cache_dir}/{self.dataname}_la.npy" if self.use_cache else None
        )
        self.la, self.la_time = self._load_or_compute_matrix(filepath, compute_la)
        print(f"  Done. Time elapsed: {self.la_time:.4f}s")

    def _build_high_order_matrix(self):
        """构建高阶结构矩阵 (ls)。"""
        print("  Building high-order graph matrix (ls)...")

        def compute_ls():
            G: sp.csr_matrix = self.adj_matrix_sparse.tocsr()
            try:
                # 尝试使用稀疏矩阵计算高阶结构矩阵
                ls: sp.csr_matrix = high_order(G, self.order, self.decay)
                ls: np.ndarray = ls.toarray().astype(array_dtype)
            except Exception as e:
                # 如果稀疏计算失败，尝试使用稠密矩阵计算
                print(f"    Warning: {e}. Trying with dense matrix...")
                G = G.toarray().astype(array_dtype)
                ls: np.ndarray = high_order(G, self.order, self.decay)

            max_val = np.diag(ls).max()
            if max_val > 0:
                ls /= max_val
            return ls

        filepath = (
            f"{self.cache_dir}/{self.dataname}_ls.npy" if self.use_cache else None
        )
        self.ls, self.ls_time = self._load_or_compute_matrix(filepath, compute_ls)
        print(f"  Done. Time elapsed: {self.ls_time:.4f}s")

    def _build_joint_matrix(self):
        """构建联合矩阵 (li)。"""
        print("  Building joint matrix (li)...")

        def compute_li():
            # 确保ls是稠密的
            ls_dense = (
                self.ls.toarray().astype(array_dtype)
                if sp.issparse(self.ls)
                else self.ls
            )

            lc = self.la @ ls_dense
            li = high_order(lc, self.order, self.decay)
            li = li.astype(array_dtype)

            max_val = np.diag(li).max()
            if max_val > 0:
                li /= max_val
            return li

        filepath = (
            f"{self.cache_dir}/{self.dataname}_li.npy" if self.use_cache else None
        )
        self.li, self.li_time = self._load_or_compute_matrix(filepath, compute_li)
        print(f"  Done. Time elapsed: {self.li_time:.4f}s")

    def _load_or_compute_matrix(
        self, filepath: str, compute_func
    ) -> tuple[np.ndarray, float]:
        """
        辅助方法：如果缓存文件存在则加载，否则计算并保存。

        Args:
            filepath (str): 缓存文件路径。
            compute_func (callable): 用于计算矩阵的函数。

        Returns:
            tuple: (计算得到的矩阵, 计算耗时)
        """
        if self.use_cache and filepath and os.path.exists(filepath):
            print(f"    Loading from cache: {filepath}")
            start_time = time.time()
            matrix = np.load(filepath)
            load_time = time.time() - start_time
            return matrix, load_time

        start_time = time.time()
        matrix = compute_func()
        elapsed_time = time.time() - start_time

        if self.use_cache and filepath:
            print(f"    Saving to cache: {filepath}")
            np.save(filepath, matrix)

            # 保存时间日志（如果需要）
            log_dir = "results/logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            time_log_path = f"{log_dir}/{self.dataname}_{os.path.basename(filepath).split('_')[1]}_time.txt"
            with open(time_log_path, "w") as f:
                f.write(f"{elapsed_time:.4f} s")

        return matrix, elapsed_time

    def _print_summary_table(self):
        """打印预处理结果的汇总表格。"""
        console = Console()
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)

        table.add_column("Matrix", style="dim")
        table.add_column("Time (s)")
        table.add_column("Type")
        table.add_column("Dtype")
        table.add_column("Memory (MB)")

        def get_matrix_info(matrix, name, time):
            return [
                name,
                f"{time:.4f}",
                f"{type(matrix).__name__}",
                f"{matrix.dtype.name}",
                f"{matrix.nbytes/(1024**2):.1f}",
            ]

        table.add_row(*get_matrix_info(self.la, "Similarity (la)", self.la_time))
        table.add_row(*get_matrix_info(self.ls, "High-order (ls)", self.ls_time))
        table.add_row(*get_matrix_info(self.li, "Joint (li)", self.li_time))

        console.print("\n[bold green]Preprocessing Summary:[/bold green]")
        console.print(table)


class Experiment:
    """
    一个用于执行多层联合非负矩阵分解(ML-JNMF)模型完整实验流程的类。
    """

    def __init__(
        self,
        dataname: str,
        pred_method: Literal["lambda", "communitude"],
        preprocessParams: Dict[str, Any],
        mfParams: Dict[str, Any],
    ):
        """
        初始化实验实例。

        Args:
            dataname (str): 数据集名称。
            pred_method (Literal["lambda", "communitude"]): 社区预测方法。
            preprocessParams (Dict[str, Any]): 数据预处理参数。
            mfParams (Dict[str, Any]): 矩阵分解模型参数。
        """
        self.dataname = dataname
        self.pred_method = pred_method
        self.preprocessParams = preprocessParams
        self.mfParams = mfParams

        # 初始化实验状态
        self.EXPERIMENT_TIME = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        self.COMPUTE_TIME_START = None
        self.PREPROCESSING_TIME = None
        self.ML_JNMF_TIME = None

        self.features_sparse = None
        self.adj_matrix_sparse = None
        self.la, self.ls, self.li = None, None, None
        self.id_and_targets = None
        self.targets = None
        self.evaluate = True
        self.comment = None

        self.r = None
        self.model = None
        self.cluster_labels = None
        self.metrics = None

        # 定义文件路径
        self.targets_path = f"stgraphs/{self.dataname}.targets"
        self.edges_path = f"stgraphs/{self.dataname}.edges"
        self.features_json_path = f"stgraphs/{self.dataname}.features"
        self.results_csv_path = f"results/clustering/{self.dataname}.csv"
        self.log_json_path = f"results/logs/{self.dataname}_{self.EXPERIMENT_TIME}.json"

    def run(self):
        """
        执行完整的实验流程。
        """
        console.print(
            f"[Experiment {self.EXPERIMENT_TIME}] Starting experiment for dataset '{self.dataname}'...",style="bold green"
        )

        self.COMPUTE_TIME_START = time.time()

        try:
            self._load_and_preprocess_data()
            self._train_model()
            if self.evaluate:
                self._evaluate()
            self._save_results()

            total_time = time.time() - self.COMPUTE_TIME_START
            print(
                f"[Experiment {self.EXPERIMENT_TIME}] Experiment completed successfully in {total_time:.2f} seconds."
            )
            print(
                f"[Experiment {self.EXPERIMENT_TIME}] Log saved to: {self.log_json_path}"
            )

            return self._get_final_log()

        except Exception as e:
            print(f"[Experiment {self.EXPERIMENT_TIME}] An error occurred: {e}")
            # 可以在这里添加错误处理逻辑，如保存不完整的日志等
            raise

    def _load_and_preprocess_data(self):
        """加载并预处理数据。"""
        print("[Step 1/4] Loading and preprocessing data...")
        start_time = time.time()

        edge_undirected = True
        self.features_sparse, self.adj_matrix_sparse = load_data(self.dataname)
        preprocessor = Preprocessor(
            self.features_sparse,
            self.adj_matrix_sparse,
            self.preprocessParams,
            edge_undirected,
            self.dataname,
        )
        self.la, self.ls, self.li = preprocessor.run()

        # if sp.issparse(self.ls):
        #     self.ls = self.ls.toarray() 修改之后的preprocess只返回np.ndarray

        self.id_and_targets = pd.read_csv(self.targets_path, header=None)
        if self.id_and_targets.shape[1] == 2:
            self.targets = self.id_and_targets.values[:, 1]
            self.id_and_targets.columns = ["id", "target"]
        elif self.id_and_targets.shape[1] > 2:
            self.targets = self.id_and_targets.values[:, 2]  # 假设目标在第三列
            self.evaluate = False
            self.comment = (
                f"目标值过多，暂时不进行评估，但是已经保存在{self.results_csv_path}"
            )
        else:
            raise ValueError("检查目标值文件格式是否正确")

        self.PREPROCESSING_TIME = time.time() - start_time
        print(
            f"[Step 1/4] Data preprocessing finished in {self.PREPROCESSING_TIME:.2f} seconds."
        )

    def _train_model(self):
        """训练ML-JNMF模型并进行预测。"""
        print("[Step 2/4] Training ML-JNMF model...")

        # 计算低维空间维度r
        # self.r = calculate_r(self.la, self.ls, self.li, self.dataname)
        self.r=10 # 先指定一下
        print(f"Calculated low-dimensional space dimension r: {self.r}")

        # 创建并训练模型
        start_time = time.time()
        self.model = ML_JNMF(self.la, self.ls, self.li, **self.mfParams)
        self.cluster_labels = self.model.fit_predict(
            self.r, self.pred_method, silent=False
        )
        self.ML_JNMF_TIME = time.time() - start_time

        print(f"[Step 2/4] Model training finished in {self.ML_JNMF_TIME:.2f} seconds.")

    def _evaluate(self):
        """评估模型性能。"""
        print("[Step 3/4] Evaluating results...")
        from src.evaluator import externalEvaluator, internalEvaluator
        from src.helpers import json2featmat

        edges = pd.read_csv(self.edges_path, header=None, sep=",")
        features = json2featmat(self.features_json_path).toarray()

        ee = externalEvaluator(self.cluster_labels, self.targets)
        ie = internalEvaluator(self.cluster_labels, edges.values, features)

        self.metrics = ee.get_all_metrics()
        self.metrics.update(ie.get_all_metrics())

        print(f"[Step 3/4] Evaluation completed. Metrics: {self.metrics}")

    def _save_results(self):
        """保存预测结果和实验日志。"""
        print("[Step 4/4] Saving results and logs...")

        # 保存预测结果
        cluster_labels_df = pd.DataFrame(self.cluster_labels, columns=["predict"])
        pred_and_targets = pd.concat([self.id_and_targets, cluster_labels_df], axis=1)

        # 写入注释和结果
        with open(self.results_csv_path, "w") as f:
            comment_lines = (
                f"# Experiment Parameters:\n"
                f"# Time: {self.EXPERIMENT_TIME}\n"
                f"# Prediction Method: {self.pred_method}\n"
                f"# Preprocessing Params: {self.preprocessParams}\n"
                f"# MF Params: {self.mfParams}\n"
                f"# r: {self.r}\n"
                f"# Comment: {self.comment if self.comment else 'N/A'}\n"
            )
            f.write(comment_lines)
        pred_and_targets.to_csv(self.results_csv_path, index=False, mode="a")

        print(f"Predictions saved to: {self.results_csv_path}")

    def _get_final_log(self):
        """生成并保存最终的实验日志。"""
        log = {
            "experiment_time": self.EXPERIMENT_TIME,
            "dataname": self.dataname,
            "predict_method": self.pred_method,
            "preprocess_params": self.preprocessParams,
            "mf_params": self.mfParams,
            "r": self.r,
            "size": getattr(
                self.model, "size", None
            ),  # 使用getattr避免模型没有size属性的错误
            "compute_time": time.time() - self.COMPUTE_TIME_START,
            "preprocessing_time": self.PREPROCESSING_TIME,
            "ml_jnmf_time": self.ML_JNMF_TIME,
            "final_loss": getattr(self.model, "final_loss", None),
            "is_converged": getattr(self.model, "is_converged", None),
            "early_stopping": getattr(self.model, "early_stopping", None),
            "evaluation": self.metrics,
            "evaluation_comment": self.comment,
        }

        with open(self.log_json_path, "w") as f:
            json.dump(log, f, indent=4)

        return log


def precompute(
    dataname: str,
    preprocessParams: dict,
):
    edge_undirected = True
    features_sparse, adj_matrix_sparse = load_data(dataname)
    # 数据预处理：生成属性层、结构层和跨层信息矩阵
    PREPROCESSING_TIME = time.time()
    preprocessor = Preprocessor(
        features_sparse,
        adj_matrix_sparse,
        preprocessParams,
        edge_undirected,
        dataname,
    )
    la, ls, li = preprocessor.run()
    PREPROCESSING_TIME = time.time() - PREPROCESSING_TIME
    with open(f"precomputed/{dataname}_preprocessing_time.txt", "w") as f:
        f.write(f"{PREPROCESSING_TIME:.4f} s")
    # r = calculate_r(la, ls, li, dataname)


def main():
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(
        description="Run ML-JNMF experiment with specified parameters"
    )

    # 添加命令行参数
    parser.add_argument(
        "--dataname",
        type=str,
        default="cora",
        help="指定要处理的数据集名称（含专属后缀，若有），可选值包括：citeseer、cora、git_web_ml、lastfm_asia、twitch_DE、twitch_ENGB、twitch_ES、twitch_FR、twitch_PTBR、twitch_RU",
    )
    parser.add_argument(
        "--pred_method",
        choices=["lambda", "communitude"],
        default=pred_method,
        help="Prediction method to use ('lambda' or 'communitude')",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=preprocessParameters["order"],
        help="Order for preprocessing, default is 5",
    )
    parser.add_argument(
        "--decay",
        type=float,
        default=preprocessParameters["decay"],
        help="Decay factor for preprocessing, default is 2",
    )
    parser.add_argument(
        "--interWeight",
        type=float,
        default=mfParameters["interWeight"],
        help="Weight for inter-layer interaction, default is 3",
    )
    parser.add_argument(
        "--pairwiseWeight",
        type=float,
        default=mfParameters["pairwiseWeight"],
        help="Weight for pairwise interaction, default is 2",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 更新预处理和矩阵分解的参数
    preprocessParams = {"order": args.order, "decay": args.decay}
    mfParams = {"interWeight": args.interWeight, "pairwiseWeight": args.pairwiseWeight}
    pred_method = args.pred_method
    dataname = args.dataname
    # 运行实验
    build_folders()
    extract_data()
    console.print(
        f"Running experiment on {dataname} with {pred_method},{preprocessParams},{mfParams}",
        style="magenta",
    )
    exp = Experiment(dataname, pred_method, preprocessParams, mfParams)
    exp.run()
    # compare(args.dataname, result)
    console.print(
        f"Experiment on {dataname} with {pred_method} completed",
        style="green",
    )

# 推荐流程：先预计算la,li,ls，再运行实验，内存占用上预计预计算会比ML_JNMF还高
if __name__ == "__main__":
    # main()
    all_datanames = []
    for file in os.listdir("stgraphs"):
        if file.endswith(".edges"):
            dataname = file.split(".")[0]
            all_datanames.append(dataname)

    for dataname in ["cora"]:
        console.print(
            f"Running experiment on {dataname} with {pred_method}, {preprocessParameters},{mfParameters}",
            style="magenta",
        )
        try:
            build_folders()
            exp = Experiment(dataname, pred_method, preprocessParameters, mfParameters)
            exp.run()
        except MemoryError as e:
            print(f"内存不足，跳过数据集 {dataname}: {str(e)}")
            continue
        except np.core._exceptions._ArrayMemoryError as e:
            print(f"NumPy数组内存错误，跳过数据集 {dataname}: {str(e)}")
            continue
        except Exception as e:
            print(f"处理数据集 {dataname} 时发生未知错误: {str(e)}")
            traceback.print_exc()
            continue
