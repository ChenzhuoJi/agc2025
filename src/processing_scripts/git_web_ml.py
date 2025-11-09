import numpy as np
import pandas as pd
import json
from src.processor import (
    edge_preprocess,
    json_to_features,
    GraphProcessingManager,
    standardize_graph,
)
from src.helpers import GraphAnalysis as ga
edges = pd.read_csv("data/raw/git_web_ml/musae_git_edges.csv")
edges.to_csv("data/graphs/git_web_ml.edges", index=False, header=False)
features = json.load(open("data/raw/git_web_ml/musae_git_features.json"))
json.dump(features, open("data/graphs/git_web_ml.features", "w"), indent=4)
targets = pd.read_csv("data/raw/git_web_ml/musae_git_target.csv")[["id", "ml_target"]]
targets.to_csv("data/graphs/git_web_ml.targets", index=False, header=False)

edges = pd.read_csv("data/raw/lasftm_asia/lastfm_asia_edges.csv", header=None)
features = json.load(open("data/raw/lasftm_asia/lastfm_asia_features.json"))
json.dump(features, open("data/graphs/lastfm_asia.features", "w"), indent=4)

targets = pd.read_csv("data/raw/lasftm_asia/lastfm_asia_target.csv")
targets.to_csv("data/graphs/lastfm_asia.targets", index=False, header=False)
all_nodes = list(features.keys())
edges = edges.astype(str)
mask = edges[0].isin(all_nodes) & edges[1].isin(all_nodes)
edges = edges[mask]
print(f"删除了{mask.size - np.sum(mask)}条边")
edges.to_csv("data/graphs/lastfm_asia.edges", index=False, header=False)
# print(len(features) == len(targets))
# features_std, edges_idx, node_to_idx = standardize_graph(
#     features,edges=edges, nodes=nodes
# )

# adjacency_matrix = edge_preprocess(edges_idx)

# print("正在进行分析")
# graph_analysis = ga(adjacency_matrix, list(targets))
# graph_analysis.comprehensive_analysis()

# start_time = time.time()
# gpm = GraphProcessingManager("git_web_ml")
# sample_size = 500
# print("正在进行处理")
# gpm.grid_search(edges, features, targets, sample_size=sample_size, overwrite=True)

# print("已完成网格搜索")
# end_time = time.time()
# print(f"处理总耗时: {end_time - start_time} 秒, sample_size={sample_size} ")
