import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np

# 读入数据
df = pd.read_csv("results/search/lasftm_asia_20251031_1116.csv")

# 取对数以防极端值影响
df["log_final_loss"] = np.log10(df["final_loss"].replace(0, np.nan))

# 要分析的指标
metrics = [
    "Accuracy",
    "Adjusted_Rand_Index",
    "Normalized_MI",
    "F1_Score",
    "log_final_loss",
]

anova_results = {}

for metric in metrics:
    model = ols(f"{metric} ~ C(p) + C(theta) + C(mu1) + C(mu2)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_results[metric] = anova_table

# 打印结果
for m, res in anova_results.items():
    print(f"\n📊 ANOVA for {m}")
    print(res[["sum_sq", "F", "PR(>F)"]])

import seaborn as sns
import matplotlib.pyplot as plt

pivot = df.pivot_table(values="F1_Score", index="p", columns="theta", aggfunc="mean")
sns.heatmap(pivot, annot=True, cmap="magma")
plt.title("Heatmap of F1_Score by p & theta")
plt.show()

from sklearn.linear_model import LinearRegression

X = df[["p", "theta", "mu1", "mu2"]]
y = df["Accuracy"]
print(LinearRegression().fit(X, y).coef_)
