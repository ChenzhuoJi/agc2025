import pandas as pd

df = pd.read_csv("results/search/lasftm_asia_20251030_2214.csv")
print(df.describe())

import matplotlib.pyplot as plt
import numpy as np

plt.hist(np.log10(df['final_loss']), bins=50)
plt.xlabel("log10(final_loss)")
plt.ylabel("count")
plt.title("Distribution of log10(final_loss)")
plt.show()

import seaborn as sns

for param in ['p', 'theta', 'mu1', 'mu2']:
    sns.boxplot(x=param, y=np.log10(df['final_loss']), data=df)
    plt.title(f"{param} vs log10(final_loss)")
    plt.show()
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols("np.log10(final_loss) ~ p + theta + mu1 + mu2", data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

pivot = df.pivot_table(values='final_loss', index='p', columns='theta', aggfunc='mean')
sns.heatmap(np.log10(pivot), annot=True, fmt=".1f")
plt.title("Heatmap of log10(final_loss) by p & theta")
plt.show()
