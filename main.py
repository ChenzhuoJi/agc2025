import argparse
import datetime
import json
from typing import Literal
import time
import zipfile
import os

import numpy as np
import pandas as pd
from rich.console import Console

from src.ml_jnmf import ML_JNMF
from src.processor import feature_process, edge_process, high_order
from src.helpers import compute_AS_with_NMF

preprocessParameters = {"order": 5, "decay": 2}

mfParameters = {"interWeight": 4, "pairwiseWeight": 2}

pred_method = "lambda"

defalut_dataname = "cora"
console = Console()


def build_folders():
    os.makedirs("results/clustering/", exist_ok=True)
    os.makedirs("results/", exist_ok=True)
    os.makedirs("results/logs/", exist_ok=True)
    os.makedirs("data/", exist_ok=True)


def extract_data():
    """从压缩包中提取文件。"""
    with zipfile.ZipFile("graphs.zip", "r") as zip_ref:
        if os.path.exists("data/graphs"):
            return
        else:
            zip_ref.extractall("data/")


def preprocess(
    dataname: str,
    preprocessParams: dict = preprocessParameters,
    edge_undirected: bool = True,
):
    order = preprocessParams["order"]
    decay = preprocessParams["decay"]

    la = feature_process(dataname)
    ls = high_order(edge_process(dataname, edge_undirected), order, decay)
    la /= np.max(la)
    ls /= np.max(ls)
    lc = la @ ls
    li = high_order(lc, order, decay)
    li /= np.max(li)
    return la, ls, li


def calculate_r(la, ls, li, targets):
    r = np.unique(targets).size
    return r


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

    # 数据预处理：生成属性层、结构层和跨层信息矩阵
    PREPROCESSING_TIME = time.time()
    la, ls, li = preprocess(dataname, preprocessParams, edge_undirected)
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

    # 计算低维空间维度r（基于数据特征和目标值）
    r = calculate_r(la, ls, li, targets)
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

    for dataname in ["cora"]:
        console.print(
            f"Running experiment on {dataname} with {pred_method}, {preprocessParameters},{mfParameters}",
            style="magenta",
        )
        try:
            experiment(dataname, pred_method, preprocessParameters, mfParameters)
        except MemoryError as e:
            print(f"内存不足，跳过数据集 {dataname}: {str(e)}")
            continue
        except np.core._exceptions._ArrayMemoryError as e:
            print(f"NumPy数组内存错误，跳过数据集 {dataname}: {str(e)}")
            continue
        except Exception as e:
            print(f"处理数据集 {dataname} 时发生未知错误: {str(e)}")
            continue
