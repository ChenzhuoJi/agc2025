import pandas as pd
import statsmodels.api as sm
df = pd.read_csv(r"results\search\lasftm_asia_20251031_1116.csv")
# 确保 is_early_stopped 为 0/1
df["is_early_stopped"] = df["is_early_stopped"].astype(int)

X = df[["p", "theta", "mu1", "mu2"]]
X = sm.add_constant(X)
y = df["is_early_stopped"]

logit_model = sm.Logit(y, X).fit()
print(logit_model.summary())
