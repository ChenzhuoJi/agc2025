import pandas as pd
import json
import numpy as np

def process_data(dataname):
    # 定义文件路径
    features_file = f"data/raw/{dataname}/{dataname}.content"
    edges_file = f"data/raw/{dataname}/{dataname}.cites"
    output_dir = "data/graphs"
    
    # 读取数据
    feature_and_targets = pd.read_csv(features_file, sep="\t", header=None, index_col=0,dtype={0:str})
    
    edges = pd.read_csv(edges_file, sep="\t", header=None)
    edges = edges.astype(str)
    
    features = feature_and_targets.iloc[:, :-1]
    targets = feature_and_targets.iloc[:, -1]
    
    # 创建映射字典
    node2idx = {node: idx for idx, node in enumerate(features.index)}
    target2idx = {target: idx for idx, target in enumerate(targets.unique())}
    targets = targets.map(target2idx).reset_index()
    targets.columns = ['id','target']
    targets['id'] = targets['id'].map(node2idx)
    targets.to_csv(f"data/graphs/{dataname}.targets", index=False,header=False)

    # 保存映射字典到文件
    json.dump(target2idx, open(f"{output_dir}/{dataname}.target2idx", "w"), indent=4)
    json.dump(node2idx, open(f"{output_dir}/{dataname}.node2idx", "w"), indent=4)
    
    # 保存特征数据
    json_data = {}
    for i, row in features.iterrows():
        feature_list = list(row.index[row == 1])
        node_idx = node2idx[i]
        json_data[node_idx] = feature_list
    json.dump(json_data, open(f"{output_dir}/{dataname}.features", "w"), indent=4)
    
    # 处理边数据
    edges.columns = ['id1', 'id2']
    
    all_nodes = features.index
    mask = edges['id1'].isin(all_nodes) & edges['id2'].isin(all_nodes)
    edges = edges[mask]
    
    print(f"删除了{mask.size - np.sum(mask)}条边")
    
    edges['id1'] = edges['id1'].map(node2idx)
    edges['id2'] = edges['id2'].map(node2idx)
    # 保存边数据
    edges.to_csv(f"{output_dir}/{dataname}.edges", index=False, header=False)
    print('边所关联的的节点数是否和特征关联的节点数相同:',np.unique(edges.values).size == features.index.size)
# 调用函数，传入数据集名称
process_data("citeseer")
process_data("cora")
