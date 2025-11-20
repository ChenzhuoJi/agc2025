import networkx as nx
import pandas as pd
import numpy as np
from main import Preprocessor, array_dtype, Experiment,preprocessParameters,mfParameters
import scipy.sparse as sp
from src.processor import feature_process
import time
import os


# 难点：由于xmark人工数据的特征不是稀疏的，所以不能直接复用main里的函数
# 规定: xmark的datanme命名格式：xmark{N}_{type_attr}
def load_artificial_data(dataname):
    datadir = r"D:\Code_Repo\.temp2\artificial"
    filename = dataname
    filepath = os.path.join(datadir, filename)
    featurefile = os.path.join(filepath + ".features")
    edgesfile = os.path.join(filepath + ".edges")

    G = nx.read_edgelist(edgesfile, delimiter=",", nodetype=int)
    adj_matrix_sparse = nx.adjacency_matrix(
        G, nodelist=sorted(G.nodes())
    )  # 这个应该改好了,节点id是对应的

    X = pd.read_csv(featurefile, sep=",", header=None, index_col=0)
    X = X.iloc[:, 1:]  # 去掉第一列（节点ID）
    feature_matrix = X.to_numpy()
    feature_matrix = feature_matrix.astype(array_dtype)

    return feature_matrix, adj_matrix_sparse


class DensePreprocessor(Preprocessor):
    """
    继承自 Preprocessor，专门用于处理非稀疏（稠密）特征矩阵的预处理类。
    """

    def __init__(
        self,
        features_matrix: np.ndarray,  # 这里改为接收稠密矩阵
        adj_matrix_sparse: sp.csr_matrix,
        preprocess_params: dict,
        edge_undirected: bool = True,
        dataname: str = None,
    ):
        """
        初始化 DensePreprocessor 实例。

        Args:
            features_matrix (np.ndarray): 稠密特征矩阵。
            adj_matrix_sparse (sp.csr_matrix): 稀疏邻接矩阵（结构层仍可能是稀疏的）。
            preprocess_params (dict): 预处理参数。
            edge_undirected (bool, optional): 图是否为无向图。默认 True。
            dataname (str, optional): 数据集名称，用于缓存文件命名。默认 None。
            feature_process_func (callable, optional): 计算属性相似度矩阵的函数。
            high_order_func (callable, optional): 计算高阶图矩阵的函数。
        """
        # 1. 调用父类的构造函数进行初始化
        # 注意：父类 __init__ 期望一个 features_sparse 参数，我们可以传一个 dummy 值，
        # 因为我们将在子类中重写 _build_similarity_matrix，不会用到它。
        # 或者，更优雅地，父类可以设计得更灵活，但为了最小化改动，我们传 None。
        super().__init__(
            features_sparse=None,  # 这个参数在子类中被忽略
            adj_matrix_sparse=adj_matrix_sparse,
            preprocess_params=preprocess_params,
            edge_undirected=edge_undirected,
            dataname=dataname,
        )

        # 2. 在子类中存储稠密的特征矩阵
        self.features_matrix = features_matrix
        if not isinstance(self.features_matrix, np.ndarray):
            raise TypeError("features_matrix 必须是一个 numpy.ndarray 类型的稠密矩阵。")

    def _build_similarity_matrix(self):
        """
        重写：从稠密特征矩阵构建属性相似度矩阵 (la)。
        """
        print("  Building similarity matrix (la) from dense features...")

        def compute_la():
            # 直接使用稠密的 self.features_matrix
            X = self.features_matrix

            # 假设 feature_process_func 可以处理稠密矩阵
            # 如果 feature_process_func 只能处理稀疏矩阵，这里需要进行适配
            la = feature_process(X, self.kernel,sparse=False)

            la = la.astype(array_dtype)
            max_val = np.diag(la).max()
            if max_val > 0:
                la /= max_val
            return la

        # 复用父类的缓存逻辑，只是计算函数不同
        filepath = (
            f"{self.cache_dir}/{self.dataname}_la_dense.npy" if self.use_cache else None
        )
        self.la, self.la_time = self._load_or_compute_matrix(filepath, compute_la)
        print(f"  Done. Time elapsed: {self.la_time:.4f}s")


class ArtificialExperiment(Experiment):
    """
    继承自 Experiment，专门用于处理人工数据集的实验类。
    """

    def __init__(self, dataname, pred_method, preprocessParams, mfParams):
        super().__init__(dataname, pred_method, preprocessParams, mfParams)
        datadir = r"D:\Code_Repo\.temp2\artificial"
        filename = dataname
        filepath = os.path.join(datadir, filename)

        self.targets_path = os.path.join(filepath + ".targets")
        self.edges_path = os.path.join(filepath + ".edges")
        self.features_json_path = None
        self.features_path = os.path.join(filepath + ".features")
        self.features_matrix = None

    def _load_and_preprocess_data(self):
        """重写父类方法，加载并预处理人工数据集。"""
        print("[Step 1/4] Loading and preprocessing data...")
        start_time = time.time()

        edge_undirected = True
        self.features_matrix, self.adj_matrix_sparse = load_artificial_data(
            self.dataname
        )
        preprocessor = DensePreprocessor(
            self.features_matrix,
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

    def _evaluate(self):
        """重写父类评估模型性能。"""
        print("[Step 3/4] Evaluating results...")
        from src.evaluator import externalEvaluator, internalEvaluator

        edges = pd.read_csv(self.edges_path, header=None, sep=",")
        features = pd.read_csv(self.features_path, header=None, sep=",").iloc[:,1:]

        ee = externalEvaluator(self.cluster_labels, self.targets)
        ie = internalEvaluator(self.cluster_labels, edges.values, features.values)

        self.metrics = ee.get_all_metrics()
        self.metrics.update(ie.get_all_metrics())

        print(f"[Step 3/4] Evaluation completed. Metrics: {self.metrics}")

if __name__ == "__main__":
    dataname = "xmark2000_continuous"
    pred_method = "lambda"
    preprocessParams = preprocessParameters
    mfParams = mfParameters

    experiment = ArtificialExperiment(
        dataname, pred_method, preprocessParams, mfParams
    )
    experiment.run()