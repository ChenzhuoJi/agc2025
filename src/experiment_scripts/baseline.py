import os

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from src.ml_jnmf import ML_JNMF

from src.evaluator import Evaluator
from src.helpers import paramsManager

# baseline
data = pd.read_csv("data/raw/LLF/Lazega-Law-Firm_nodes.txt", sep=r"\s+", header=0)
print(data.head(5))
X = data.drop(columns=["nodeOffice","nodeID"])
y = data["nodeOffice"]
print(X.columns)
cluster_labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
eva = Evaluator(cluster_labels, y)
eva.print_metrics()
