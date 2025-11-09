import json
import os
import time

import pandas as pd

from src.processor import GraphProcessingManager, json_to_features,edge_preprocess
from src.helpers import GraphAnalysis as ga


data_name = "lasftm_asia"
data_dir = os.path.join("data", "raw", data_name)

edges_file = os.path.join(data_dir, "lastfm_asia_edges.csv")
features_json_file = os.path.join(data_dir, "lastfm_asia_features.json")
targets_file = os.path.join(data_dir, "lastfm_asia_target.csv")

edges = pd.read_csv(edges_file)
edges.columns = ['node_1','node_2']
features = json_to_features(features_json_file)
targets = pd.read_csv(targets_file)["target"].to_numpy()
edges.to_csv("data/graphs/lasftm_asia.edges", index=False)

features.to_csv("data/graphs/lasftm_asia.features", index=False)
# adjacency_matrix, node_to_index = edge_preprocess(edges)
# print("正在进行分析")
# graph_analysis = ga(adjacency_matrix, list(targets))
# graph_analysis.comprehensive_analysis()

# start_time = time.time()
# gpm = GraphProcessingManager(data_name)
# sample_size = 500
# print("正在进行处理")
# gpm.grid_search(edges, features, targets, sample_size=sample_size, overwrite=True)

# print("已完成网格搜索")
# end_time = time.time()
# print(f"处理总耗时: {end_time - start_time} 秒, sample_size={sample_size} ")
