import os

import numpy as np
import pandas as pd

from evaluator import Evaluator
from experiment import Experiment, testExperiment, searchExperiment
from ml_jnmf import ML_JNMF

# targets = np.load(r"data\interimediate\lasftm_asia\target.npy")
# r = len(np.unique(targets))
adj_matrices = []
for i in range(1,4):
    adj_matrices.append(np.load(f'data/interimediate/LLF/adj_matrix{i}.npy'))

features=pd.read_csv(r'data\processed\LLF\features.csv')

O=features['nodeOffice'].to_numpy()
S=features['nodeStatus'].to_numpy()

model = ML_JNMF()
os.makedirs('results/test',exist_ok=True)
def mapping(row):
    if row['type'] == 'inter':
        prefix = 100
    else:
        prefix = 200
    return prefix + row['community_id']
best_nmi1 = 0
best_nmi2 = 0

for r in range(2,21):
    model = ML_JNMF()
    model.fit(adj_matrices[0],adj_matrices[1],adj_matrices[2],r=r)
    df1,df2 = model.predict()
    # 将 （层内or层间 数据转换为 数字数据）
    cluser_labels_1 = df1.apply(mapping,axis=1).to_numpy()
    cluser_labels_2 = df2.apply(mapping,axis=1).to_numpy()
    eva1 = Evaluator(cluser_labels_1,O)
    eva2 = Evaluator(cluser_labels_2,O)
    nmi1 = eva1.normalized_mutual_information()

    nmi2 = eva2.normalized_mutual_information()
    best_nmi1 = max(best_nmi1, nmi1)
    best_nmi2 = max(best_nmi2, nmi2)
print(best_nmi1)
print(best_nmi2)
#         result_layer1=create_mapping(df1)['mapping'].to_numpy()
#         result_layer2=create_mapping(df2)['mapping'].to_numpy()
#         eva_O1=evaluate_all(O,result_layer1)
#         eva_S1=evaluate_all(S,result_layer1)
#         eva_O2=evaluate_all(O,result_layer2)
#         eva_S2=evaluate_all(S,result_layer2)

#         if max(eva_O1['NMI'],eva_O2['NMI'])>best_NMI:
#             best_NMI=max(eva_O1['NMI'],eva_O2['NMI'])
#             best_r1, best_r2 = r1, r2

# print(f"best_r1:{best_r1}, best_r2:{best_r2},best_NMI:{best_NMI}")

# epr = searchExperiment("LLF", 10)
# res = epr.run()

# epr.run_test()
# # layer1
# # ho_matrix = np.load(os.path.join(MAT_PATH, 'ho_matrix1.npy'))
# # similarity_matrix = np.load(os.path.join(MAT_PATH, 'similarity_matrix.npy'))
# # adj_matrix = np.load(os.path.join(MAT_PATH, 'adj_matrix1.npy'))
# # inter_matrix = (ho_matrix + similarity_matrix)/2

# # for r1 in tqdm(range(1,20)):
# #     for r2 in range(1,20):
# #         model = ML_JNMF()
# #         model.fit(intra1, intra2, inter, r1,r2)
# #         best_loss= np.inf
# #         if model.early_stopper.best_loss < best_loss:
# #             best_loss = model.early_stopper.best_loss
# #             best_r1, best_r2 = r1, r2
# model=ML_JNMF()
# model.fit(intra1,intra2,inter,6,7)
# df1,df2=model.predict()


# O=features['nodeOffice'].to_numpy()
# S=features['nodeStatus'].to_numpy()
# best_NMI=0

# for r1 in tqdm(range(2,21)):
#     for r2 in range(1,20):
#         model = ML_JNMF()
#         model.fit(intra1, intra2, inter, r1,r2)
#         label=model.predict()
#         result_layer1=create_mapping(df1)['mapping'].to_numpy()
#         result_layer2=create_mapping(df2)['mapping'].to_numpy()
#         eva_O1=evaluate_all(O,result_layer1)
#         eva_S1=evaluate_all(S,result_layer1)
#         eva_O2=evaluate_all(O,result_layer2)
#         eva_S2=evaluate_all(S,result_layer2)

#         if max(eva_O1['NMI'],eva_O2['NMI'])>best_NMI:
#             best_NMI=max(eva_O1['NMI'],eva_O2['NMI'])
#             best_r1, best_r2 = r1, r2

# print(f"best_r1:{best_r1}, best_r2:{best_r2},best_NMI:{best_NMI}")
