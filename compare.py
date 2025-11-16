import pandas as pd
import numpy as np
import os
from src.helpers import json2featmat
from src.evaluator import externalEvaluator,internalEvaluator
def compare(dataname):
    master_file = f'results/clustering/{dataname}.csv'
    baseline_file = f'results/baseline/{dataname}.csv'

    master_df = pd.read_csv(master_file,comment='#')[['id','predict']]
    master_df.columns = ['id','master']
    baseline_df = pd.read_csv(baseline_file)
     # 合并两个DataFrame，根据node_id对齐
    merged_df = pd.merge(master_df, baseline_df, on='id', suffixes=('_master', '_baseline'))
    merged_df.to_csv(f"results/compare/{dataname}.csv", index=False)
def evaluate(dataname):
    compare_file = f"results/compare/{dataname}.csv"
    df = pd.read_csv(compare_file,header=0,sep=',')
    reference_labels = df['target'].values
    algorithms = []
    for col in df.columns:
        if col not in ['id','target']:
            algorithms.append(col)
    mertrics = {}
    for algorithm in algorithms:
         
        cluster_labels = df[algorithm].values
        external = externalEvaluator(cluster_labels,reference_labels)
        edges = pd.read_csv(rf"stgraphs\{dataname}.edges", header=None, sep=",").values
        features = json2featmat(f"stgraphs/{dataname}.features").toarray()

        internal = internalEvaluator(cluster_labels,edges,features)
        mertric = external.get_all_metrics()
        mertric.update(internal.get_all_metrics())
        mertrics[algorithm] = mertric
        
    result_df = pd.DataFrame(mertrics)
    result_df_reset = result_df.reset_index()
    result_df_reset.rename(columns={'index': 'metric'}, inplace=True)
    
    # 保存到CSV文件
    result_df_reset.to_csv(f"results/compare/{dataname}_metrics.csv", index=False)
if __name__ == '__main__':
    os.makedirs("results/compare", exist_ok=True)
    dataname = 'cora'
    compare(dataname)
    evaluate(dataname)

