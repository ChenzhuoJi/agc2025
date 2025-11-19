import argparse
import datetime
import json
from typing import Literal
import time
import zipfile
import os
import traceback

import numpy as np
import pandas as pd
import scipy.sparse as sp
import igraph as ig
import leidenalg
from sklearn.preprocessing import normalize
from rich.console import Console
from rich.table import Table
from rich import box

import networkx as nx

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
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
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
    adj_matrix_sparse = sp.csr_matrix((data, (row, col)), shape=(n, n))
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


# TODO: 优化preprocess，主要优化联合矩阵的计算时间
# 方案一：预处理后，全都转化为csr_matrix，非常慢 ❌
# 方案二：预处理后，全都转化为np.ndarray ✅，citeseer:2.1816s, 但是大矩阵的高阶运算时间还是会很大。最大占用内存约20GB，在处理30000节点的数据时出现
# 进一步优化，发现ls稀疏性并不是都很好，内存占用有时候比稠密矩阵要大，全都转换为np.ndarray
def preprocess(
    features_sparse: sp.csr_matrix,
    adj_matrix_sparse: sp.csr_matrix,
    preprocessParams: dict = preprocessParameters,
    edge_undirected: bool = True,
    dataname: str = None
):
    kernel = preprocessParams["feature_kernel"]
    order = preprocessParams["order"]
    decay = preprocessParams["decay"]
    la_time = 0
    ls_time = 0
    li_time = 0
    if dataname is not None:
        read = True
    
    t1 = time.time()

    if read == True and os.path.exists(f"precomputed/{dataname}_la.npy"):
        la = np.load(f"precomputed/{dataname}_la.npy")
    else:
        X = features_sparse.tocsr()
        la = feature_process(X, kernel)  # 这个矩阵是稠密的np.ndarray
        la = la.astype(array_dtype)
        max_val = np.diag(la).max()
        if max_val > 0:
            la /= max_val
        t2 = time.time()
        la_time = t2 - t1
        with open(f"results/logs/{dataname}_la_time.txt", "w") as f:
            f.write(f"{la_time:.4f} s")
        np.save(f"precomputed/{dataname}_la.npy", la)

    t3 = time.time()
    if read == True and os.path.exists(f"precomputed/{dataname}_ls.npy"):
        ls = np.load(f"precomputed/{dataname}_ls.npy")
    else:
        G = adj_matrix_sparse.tocsr()
        ls = high_order(G, order, decay)  # 这个矩阵是稀疏的sp.csr_matrix
        ls = ls.toarray().astype(array_dtype)
        max_val = np.diag(ls).max()
        if max_val > 0:
            ls /= max_val
        t4 = time.time()
        ls_time = t4 - t3
        with open(f"results/logs/{dataname}_ls_time.txt", "w") as f:
            f.write(f"{ls_time:.4f} s")
        np.save(f"precomputed/{dataname}_ls.npy", ls)

    t5 = time.time()
    if read == True and os.path.exists(f"precomputed/{dataname}_li.npy"):
        li = np.load(f"precomputed/{dataname}_li.npy")
    else:
        # 转换ls为np.ndarray，适应后续与la的矩阵乘法
        if sp.issparse(ls):
            ls_array = ls.toarray()

        # 计算lc，这是一个稠密的np.ndarray。
        lc = la @ ls_array
        # 计算li，这是一个稠密的np.ndarray。
        li = high_order(lc, order, decay)
        max_val = np.diag(li).max()
        if max_val > 0:
            li /= max_val
        t6 = time.time()

        li_time = t6 - t5
        with open(f"results/logs/{dataname}_li_time.txt", "w") as f:
            f.write(f"{li_time:.4f} s")
        np.save(f"precomputed/{dataname}_li.npy", li)

    # la,li是稠密矩阵，ls是稀疏矩阵
    table = Table(show_header=True, header_style="default", box=box.SIMPLE)

    # 移除列的颜色样式
    table.add_column("矩阵")
    table.add_column("处理时间")
    table.add_column("数据格式")
    table.add_column("内存占用")

    # 添加数据行，保持内容不变
    table.add_row(
        "相似度矩阵",
        f"{la_time:.4f} s",
        f"{type(la).__name__} {la.dtype.name}",
        f"{la.nbytes/(1024**2):.1f} MB",
    )
    table.add_row(
        "高阶图矩阵",
        f"{ls_time:.4f} s",
        f"{type(ls).__name__} {ls.dtype}",
        f"{ls.nbytes/(1024**2):.1f} MB",
    )
    table.add_row(
        "联合图矩阵",
        f"{li_time:.4f} s",
        f"{type(li).__name__} {li.dtype.name}",
        f"{li.nbytes/(1024**2):.1f} MB",
    )

    console.print(table)

    return la, ls, li


# TODO: 把论文中的calculate_r实现过来
def calculate_r(la, ls, li, dataname):
    # 自动确定最优社区数量
    if os.path.exists(f"results/best_r/{dataname}.txt"):
        with open(f"results/best_r/{dataname}.txt", "r") as f:
            best_r = int(f.read().strip())
    else:
        best_r = determine_community_number(li, max_r=10, save_path=f"results/best_r/{dataname}")
        with open(f"results/best_r/{dataname}.txt", "w") as f:
            f.write(f"{best_r}")
    return best_r


def experiment(
    dataname: str,
    pred_method: Literal["lambda", "communitude"],
    preprocessParams: dict,
    mfParams: dict,
):
    """执行多层联合非负矩阵分解(ML-JNMF)模型的完整实验流程。"""
    # 记录实验开始时间（用于文件名标识）
    EXPERIEMENT_TIME = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    # 记录计算开始时间（用于统计总计算时间）
    COMPUTE_TIME = time.time()

    edge_undirected = True
    features_sparse, adj_matrix_sparse = load_data(dataname)
    # 数据预处理：生成属性层、结构层和跨层信息矩阵
    PREPROCESSING_TIME = time.time()
    la, ls, li = preprocess(
        features_sparse, adj_matrix_sparse, preprocessParams, edge_undirected, dataname
    )
    if sp.issparse(ls):
        ls = ls.toarray()
    PREPROCESSING_TIME = time.time() - PREPROCESSING_TIME
    id_and_targets = pd.read_csv(f"stgraphs/{dataname}.targets", header=None)

    evaluate = True
    if id_and_targets.shape[1] == 2:
        targets = id_and_targets.values[:, 1]
        id_and_targets.columns = ["id", "target"]
        comment = None

    elif id_and_targets.shape[1] > 2:
        targets = id_and_targets.values[:, 2]
        evaluate = False
        comment = f"目标值过多，暂时不进行评估，但是已经保存在results/clustering/{dataname}.csv"
    else:
        raise ValueError("检查目标值是否正确")

    # 计算低维空间维度r
    r = calculate_r(la, ls, li, dataname)
    print(f"r: {r}")

    # 创建ML-JNMF模型实例并传入数据和模型参数
    model = ML_JNMF(la, ls, li, **mfParams)

    # 模型训练并进行预测，返回预测的社区结果
    ML_JNMF_TIME = time.time()
    cluster_labels = model.fit_predict(r, pred_method, silent=False)
    ML_JNMF_TIME = time.time() - ML_JNMF_TIME
    if evaluate:
        from src.evaluator import externalEvaluator, internalEvaluator
        from src.helpers import json2featmat

        edges = pd.read_csv(f"stgraphs/{dataname}.edges", header=None, sep=",")
        features = json2featmat(f"stgraphs/{dataname}.features").toarray()

        ee = externalEvaluator(cluster_labels, targets)
        ie = internalEvaluator(cluster_labels, edges.values, features)
        metrics = ee.get_all_metrics()
        metrics.update(ie.get_all_metrics())

    else:
        metrics = None

    cluster_labels = pd.DataFrame(cluster_labels, columns=["predict"])
    pred_and_targets = pd.concat([id_and_targets, cluster_labels], axis=1)
    pred_and_targets_comment = (
        f"# Experiment Parameters:\n"
        f"# Preprocessing Params: {preprocessParams}\n"
        f"# MF Params: {mfParams}\n"
    )

    # 将实验参数写入CSV文件
    with open(f"results/clustering/{dataname}.csv", "w") as f:
        f.write(pred_and_targets_comment)
    pred_and_targets.to_csv(
        f"results/clustering/{dataname}.csv",
        index=False,
        mode="a",
    )

    # 计算总计算时间
    COMPUTE_TIME = time.time() - COMPUTE_TIME

    # 创建实验日志，包含实验配置、结果和性能指标
    log = {
        "dataname": dataname,  # 数据集名称
        "r": r,  # 低维空间维度
        "size": model.size,  # 数据规模（节点数量）
        "datetime": EXPERIEMENT_TIME,  # 实验时间戳
        "compute_time": COMPUTE_TIME,  # 总计算时间（秒）
        "preprocessing_time": PREPROCESSING_TIME,  # 预处理时间（秒）
        "ml_jnmf_time": ML_JNMF_TIME,
        "final_loss": model.final_loss,  # 模型最终损失值
        "is_converged": model.is_converged,  # 模型是否收敛
        "early_stopping": model.early_stopping,  # 是否触发早停机制
        "evaluation": metrics,  # 评估指标结果
        "evaluation_comments": comment,  # 评估指标结果注释
        "predict_method": pred_method,  # 使用的预测方法
    }

    # 将预处理参数和模型参数添加到日志中
    log.update(preprocessParams)
    log.update(mfParams)

    # 将日志保存为JSON文件，便于后续实验结果分析和复现
    json.dump(
        log, open(f"results/logs/{dataname}_{EXPERIEMENT_TIME}.json", "w"), indent=4
    )
    console.print(
        f"实验日志已保存到 results/logs/{dataname}_{EXPERIEMENT_TIME}.json",
        style="magenta",
    )
    return log


def precompute(
    dataname: str,
    preprocessParams: dict,
):
    edge_undirected = True
    features_sparse, adj_matrix_sparse = load_data(dataname)
    # 数据预处理：生成属性层、结构层和跨层信息矩阵
    PREPROCESSING_TIME = time.time()
    la, ls, li = preprocess(
        features_sparse, adj_matrix_sparse, preprocessParams, edge_undirected, dataname
    )
    ls = ls.toarray()
    PREPROCESSING_TIME = time.time() - PREPROCESSING_TIME
    with open(f"precomputed/{dataname}_preprocessing_time.txt", "w") as f:
        f.write(f"{PREPROCESSING_TIME:.4f} s")
    r = calculate_r(la, ls, li, dataname)
    

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
    result = experiment(dataname, pred_method, preprocessParams, mfParams)
    # compare(args.dataname, result)
    console.print(
        f"Experiment on {dataname} with {pred_method} completed",
        style="green",
    )


if __name__ == "__main__":
    # main()
    all_datanames = []
    for file in os.listdir("stgraphs"):
        if file.endswith(".edges"):
            dataname = file.split(".")[0]
            all_datanames.append(dataname)

    for dataname in ['cora']:
        console.print(
            f"Running experiment on {dataname} with {pred_method}, {preprocessParameters},{mfParameters}",
            style="magenta",
        )
        try:
            build_folders()
            experiment(dataname, pred_method, preprocessParameters, mfParameters)
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
