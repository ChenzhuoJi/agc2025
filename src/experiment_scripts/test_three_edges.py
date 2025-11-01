import os

import pandas as pd
import numpy as np

from src.ml_jnmf import ML_JNMF
from src.evaluator import Evaluator
from src.helpers import paramsManager
df_edges = pd.read_csv(
    r"data\raw\LLF\Lazega-Law-Firm_multiplex.edges", sep=" ", header=None
)
df_edges.columns = ["layer", "node1", "node2", "weight"]

node_ids = np.unique(df_edges[["node1", "node2"]].values)
# 构造邻接矩阵
adj_matrices = []
for layer in df_edges["layer"].unique():
    df_layer = df_edges[df_edges["layer"] == layer]
    adj_matrix = np.zeros((len(node_ids), len(node_ids)))
    for _, row in df_layer.iterrows():
        node1, node2, weight = row["node1"], row["node2"], row["weight"]
        adj_matrix[node_ids.tolist().index(node1), node_ids.tolist().index(node2)] = (
            weight
        )
        adj_matrix[node_ids.tolist().index(node2), node_ids.tolist().index(node1)] = (
            weight
        )
    adj_matrices.append(adj_matrix)

advice = adj_matrices[0]
friendship = adj_matrices[1]
co_work = adj_matrices[2]

targets = pd.read_csv("data/raw/LLF/Lazega-Law-Firm_nodes.txt", sep=r"\s+", header=0)[
    "nodeOffice"
].to_numpy()

r = np.unique(targets).size
print(r)

results1_path = 'src/experiment_scripts/results1.csv'
results2_path = 'src/experiment_scripts/results2.csv'
if os.path.exists(results1_path):
    os.remove(results1_path)
if os.path.exists(results2_path):
    os.remove(results2_path)

pm = paramsManager()
for mu1 in pm.mu1_to_select:
    for mu2 in pm.mu2_to_select:
        model = ML_JNMF(mu1,mu2)
        df1,df2 = model.fit_predict(advice,co_work,friendship,r,pred_method=None)
        eva = Evaluator(df1['community_id'],targets)
        result = {}
        result['mu1'] = mu1
        result['mu2'] = mu2
        result.update(eva.get_all_metrics())
        result = pd.DataFrame(result,index=[0])
        result.to_csv(results1_path,index=False,mode='a',header=not pd.io.common.file_exists(results1_path))

        eva = Evaluator(df2['community_id'],targets)
        result = {}
        result['mu1'] = mu1
        result['mu2'] = mu2
        result.update(eva.get_all_metrics())
        result = pd.DataFrame(result,index=[0])
        result.to_csv(results2_path,index=False,mode='a',header=not pd.io.common.file_exists(results2_path))

# baseline
features = pd.read_csv("data/raw/LLF/Lazega-Law-Firm_nodes.txt", sep=r"\s+", header=0)
