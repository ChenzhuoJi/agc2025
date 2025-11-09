import pandas as pd

df = pd.read_csv(r"results\search\lasftm_asia_20251101_1646.csv")
print(df.describe())

import matplotlib.pyplot as plt
import numpy as np

# plt.hist(df['Accuracy'], bins=50)
# plt.xlabel("Accuracy")
# plt.ylabel("count")
# plt.title("Distribution of Accuracy")
# plt.show()

import seaborn as sns

# 创建 1行4列 的子图
fig, axes = plt.subplots(2,2, figsize=(12,12))  # figsize 可以调整整体宽高

# 循环绘制每个参数的箱线图
for i, param in enumerate(['p', 'theta', 'mu1', 'mu2']):
    sns.boxplot(x=param, y='NMI', data=df, ax=axes[i%2][i//2])
    axes[i%2][i//2].set_title(f"{param} vs NMI")
    axes[i%2][i//2].set_xlabel(param)
    axes[i%2][i//2].set_ylabel('NMI')

plt.tight_layout()  # 避免标题重叠
plt.show()

import statsmodels.api as sm
from statsmodels.formula.api import ols

# 检查每个参数是否对结果有显著影响
for param in ['p', 'theta', 'mu1', 'mu2']:
    model = ols(f"NMI ~ {param}", data=df).fit()
    print(f"{param} p-value: {model.pvalues[param]:.4f}")

print(df.sort_values(by='NMI', ascending=False).head(10))

# model = ols("np.log10(final_loss) ~ p + theta + mu1 + mu2", data=df).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# print(anova_table)

# pivot = df.pivot_table(values='final_loss', index='p', columns='theta', aggfunc='mean')
# sns.heatmap(np.log10(pivot), annot=True, fmt=".1f")
# plt.title("Heatmap of log10(final_loss) by p & theta")
# plt.show()
