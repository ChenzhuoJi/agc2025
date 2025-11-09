import datetime
import json
from typing import Literal
import time

import numpy as np
import pandas as pd

from src.ml_jnmf import ML_JNMF
from src.processor import feature_process, edge_process, high_order

preprocessParameters = {"order": 5, "decay": 2}

mfParameters = {"interWeight": 3, "pairwiseWeight": 2}

def preprocess(dataname: str, preprocessParams: dict = preprocessParameters):
    order = preprocessParams["order"]
    decay = preprocessParams["decay"]

    la = feature_process(dataname)

    ls = high_order(edge_process(dataname), order, decay)

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
    preprocessParams: dict = preprocessParameters,
    mfParams: dict = mfParameters,
):
    """执行多层联合非负矩阵分解(ML-JNMF)模型的完整实验流程。

    该函数负责从数据预处理开始，到模型训练、预测、评估和结果记录的全流程管理，
    适用于不同数据集和不同参数组合的实验场景。

    Args:
        dataname (str): 数据集名称，用于指定要使用的图数据
        pred_method (Literal["lambda", "communitude"]): 预测方法，
            "lambda"表示使用lambda权重加和，"communitude"表示使用社群度检测
        preprocessParams (dict): 数据预处理参数配置，默认为preprocessParameters
        mfParams (dict): 矩阵分解模型参数配置，默认为mfParameters
    """
    # 记录实验开始时间（用于文件名标识）
    EXPERIEMENT_TIME = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    # 记录计算开始时间（用于统计总计算时间）
    COMPUTE_TIME = time.time()

    # 数据预处理：生成属性层、结构层和跨层信息矩阵
    la, ls, li = preprocess(dataname, preprocessParams)

    # 读取目标值（通常是节点的真实标签/社区信息）
    targets = pd.read_csv(f"data/graphs/{dataname}.targets", header=None).values[:, 1]

    # 计算低维空间维度r（基于数据特征和目标值）
    r = calculate_r(la, ls, li, targets)
    print(f"r: {r}")

    # 创建ML-JNMF模型实例并传入数据和模型参数
    model = ML_JNMF(la, ls, li, **mfParams)

    # 模型训练并进行预测，返回预测的社区结果
    pred = model.fit_predict(r, pred_method, silent=False)

    # 评估模型性能，返回各项指标（如NMI、ARI等）
    metrics = model.evaluate(targets)

    # 将预测结果和真实目标值合并为DataFrame（便于后续分析）
    pred_and_targets = pd.DataFrame(
        np.concatenate((pred, targets.reshape(-1, 1)), axis=1),
        columns=["predict", "target"],
    )

    # 计算总计算时间
    COMPUTE_TIME = time.time() - COMPUTE_TIME

    # 创建实验日志，包含实验配置、结果和性能指标
    log = {
        "dataname": dataname,  # 数据集名称
        "r": r,  # 低维空间维度
        "size": la.shape[0],  # 数据规模（节点数量）
        "time": EXPERIEMENT_TIME,  # 实验时间戳
        "compute_time": COMPUTE_TIME,  # 总计算时间（秒）
        "final_loss": model.final_loss,  # 模型最终损失值
        "is_converged": model.is_converged,  # 模型是否收敛
        "evalution": metrics,  # 评估指标结果
        "predict_method": pred_method,  # 使用的预测方法
    }

    # 将预处理参数和模型参数添加到日志中
    log.update(preprocessParams)
    log.update(mfParams)

    # 将日志保存为JSON文件，便于后续实验结果分析和复现
    json.dump(
        log, open(f"results/logs/{dataname}_{EXPERIEMENT_TIME}.json", "w"), indent=4
    )


if __name__ == "__main__":
    experiment("cora", "lambda")
    experiment("citeseer", "lambda")
    # experiment("git_web_ml", "lambda")
