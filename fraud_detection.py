"""
Credit card fraud detection: linear algebra + statistics + anomaly detection.
Run: python fraud_detection.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
os.makedirs("figures", exist_ok=True)
SEED = 42

# =====================================================================
# 1. LOAD AND CHECK
# =====================================================================
df = pd.read_csv("data/fraud_data.csv", parse_dates=["Date"])
df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.hour
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
df["LogAmount"] = np.log(df["Amount"])
y = df["Is_Fraud"].to_numpy()

print("Shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()), " Duplicate rows:", int(df.duplicated().sum()))
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("Cards:", df["Card_Number"].nunique(), " Merchants:", df["Merchant"].nunique())
print("\nClass balance:\n", df["Is_Fraud"].value_counts())
print("Fraud rate: %.1f%%" % (100 * y.mean()))
print("Accuracy of predicting 'never fraud': %.1f%%" % (100 * (1 - y.mean())))

# =====================================================================
# 2. EXPLORATORY ANALYSIS
# =====================================================================
print("\nAmount by class:\n", df.groupby("Is_Fraud")["Amount"].describe().round(1))
max_legit = df.loc[df.Is_Fraud == 0, "Amount"].max()
print("\nLargest legitimate amount: %.2f" % max_legit)
print("Fraud cases above that amount: %d of %d" % ((df.loc[df.Is_Fraud == 1, "Amount"] > max_legit).sum(), y.sum()))
print("Transactions above 1000: %d, of which fraud: %d" % (
    (df.Amount > 1000).sum(), df.loc[df.Amount > 1000, "Is_Fraud"].sum()))

print("\nFraud rate by transaction type:\n",
      df.groupby("Transaction_Type")["Is_Fraud"].agg(["sum", "count", "mean"]).round(3))
print("\nFraud rate by day of week:\n", df.groupby("DayOfWeek")["Is_Fraud"].mean().round(3).to_dict())

# Chi-square tests: is fraud related to the categorical variables?
for col in ["Transaction_Type", "Merchant", "DayOfWeek"]:
    chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[col], df["Is_Fraud"]))
    print("Chi-square fraud vs %-17s chi2=%6.2f p=%.3f" % (col, chi2, p))
hour_f = df.loc[df.Is_Fraud == 1, "Hour"]; hour_l = df.loc[df.Is_Fraud == 0, "Hour"]
print("Hour of day, fraud vs legitimate (Mann-Whitney): p=%.3f" % stats.mannwhitneyu(hour_f, hour_l).pvalue)
u = stats.mannwhitneyu(df.loc[df.Is_Fraud == 1, "Amount"], df.loc[df.Is_Fraud == 0, "Amount"])
print("Amount, fraud vs legitimate (Mann-Whitney): p=%.2e" % u.pvalue)

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(data=df, x="Amount", hue="Is_Fraud", log_scale=True, bins=40, ax=ax[0], stat="count", common_norm=False)
ax[0].set_title("Amount by class (log scale)")
sns.barplot(data=df, x="Transaction_Type", y="Is_Fraud", errorbar=None, ax=ax[1])
ax[1].set_title("Fraud rate by transaction type"); ax[1].tick_params(axis="x", rotation=20)
sns.boxplot(data=df, x="Is_Fraud", y="Amount", ax=ax[2]); ax[2].set_yscale("log"); ax[2].set_title("Amount by class")
plt.tight_layout(); plt.savefig("figures/eda.png", dpi=150); plt.close()

# =====================================================================
# 3. LINEAR ALGEBRA: TRANSACTIONS AS VECTORS AND MATRICES
# =====================================================================
# Card deviation: log amount minus the card's average log amount over its OTHER transactions
g = df.groupby("Card_Number")["LogAmount"]
n_card, s_card = g.transform("count"), g.transform("sum")
loo_mean = ((s_card - df["LogAmount"]) / (n_card - 1)).where(n_card > 1, df["LogAmount"].mean())
df["CardDeviation"] = df["LogAmount"] - loo_mean

types = pd.get_dummies(df["Transaction_Type"]).astype(float)
df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)
feature_matrix = pd.concat([df[["LogAmount", "HourSin", "HourCos", "IsWeekend", "CardDeviation"]], types], axis=1)

X = feature_matrix.to_numpy(float)                 # n x p matrix: each ROW is one transaction vector
print("\nTransaction matrix X: shape", X.shape, "(rows = transactions, columns = features)")
print("Columns:", list(feature_matrix.columns))
mu = X.mean(axis=0)
Z = (X - mu) / X.std(axis=0, ddof=1)               # standardised matrix
C = (Z.T @ Z) / (len(Z) - 1)                       # covariance (= correlation) matrix, p x p
eigvals, eigvecs = np.linalg.eigh(C)
eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
print("\nEigenvalues of the covariance matrix:", np.round(eigvals, 3))
print("Proportion of variance explained:", np.round(eigvals / eigvals.sum(), 3))
print("Cumulative:", np.round(np.cumsum(eigvals) / eigvals.sum(), 3))

plt.figure(figsize=(6, 4))
plt.bar(range(1, len(eigvals) + 1), eigvals / eigvals.sum())
plt.xlabel("Principal component"); plt.ylabel("Variance explained"); plt.title("Scree plot")
plt.tight_layout(); plt.savefig("figures/scree.png", dpi=150); plt.close()

# Mahalanobis distance: d_i = sqrt((x_i - mu)^T C^-1 (x_i - mu)); pinv guards against the dummy-variable redundancy
Cinv = np.linalg.pinv(C)
mahal = np.sqrt(np.einsum("ij,jk,ik->i", Z, Cinv, Z))
# PCA reconstruction error using the top-k eigenvectors (via SVD)
U_, S_, Vt = np.linalg.svd(Z, full_matrices=False)
def recon_error(k):
    V = Vt[:k].T
    return ((Z - Z @ V @ V.T) ** 2).sum(axis=1)

# =====================================================================
# 4. ANOMALY DETECTION (UNSUPERVISED: labels are used ONLY to evaluate)
# =====================================================================
med = df["LogAmount"].median()
mad = stats.median_abs_deviation(df["LogAmount"], scale="normal")
df["RobustZ"] = (df["LogAmount"] - med) / mad
amount_only = StandardScaler().fit_transform(df[["LogAmount"]])
Zs = StandardScaler().fit_transform(feature_matrix)

scores = {
    "Robust z-score (log amount, MAD)": df["RobustZ"].to_numpy(),
    "Card deviation (vs own history)": df["CardDeviation"].to_numpy(),
    "Mahalanobis (all features)": mahal,
    "PCA reconstruction error (k=2)": recon_error(2),
    "Isolation Forest (all features)": -IsolationForest(n_estimators=300, random_state=SEED).fit(Zs).score_samples(Zs),
    "Isolation Forest (amount only)": -IsolationForest(n_estimators=300, random_state=SEED).fit(amount_only).score_samples(amount_only),
    "Local Outlier Factor (all features)": -LocalOutlierFactor(n_neighbors=20).fit(Zs).negative_outlier_factor_,
}
K = int(y.sum())
rows = []
for name, s in scores.items():
    top = np.argsort(-s)[:K]
    rows.append([name, roc_auc_score(y, s), average_precision_score(y, s), y[top].sum() / K])
anom = pd.DataFrame(rows, columns=["Method", "ROC_AUC", "Avg_Precision", f"Precision@{K}"])
print("\n" + "=" * 70 + "\nANOMALY DETECTION (no labels used for fitting)\n" + "=" * 70)
print(anom.round(3).to_string(index=False))

plt.figure(figsize=(7, 5))
for name, s in scores.items():
    fpr, tpr, _ = roc_curve(y, s)
    plt.plot(fpr, tpr, label="%s (%.2f)" % (name, roc_auc_score(y, s)))
plt.plot([0, 1], [0, 1], "--", color="grey")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Anomaly detection ROC")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("figures/anomaly_roc.png", dpi=150); plt.close()

# =====================================================================
# 5. SUPERVISED MODELS (repeated stratified cross-validation)
# =====================================================================
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
base = pd.concat([df[["LogAmount", "HourSin", "HourCos", "IsWeekend"]], types], axis=1)
merch = pd.get_dummies(df["Merchant"]).astype(float)
sets = {
    "Amount only": df[["LogAmount"]],
    "Amount + time + type": base,
    "Amount + time + type + merchant": pd.concat([base, merch], axis=1),
}
def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
def forest():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=SEED)

def cv_scores(make_model, Xdf):
    aucs, aps = [], []
    Xv = Xdf.to_numpy(float)
    for tr, te in cv.split(Xv, y):
        m = make_model().fit(Xv[tr], y[tr])
        p = m.predict_proba(Xv[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
    return np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)

res_rows = []
for sname, Xdf in sets.items():
    for mname, mk in [("Logistic regression", logreg), ("Random forest", forest)]:
        a, asd, p, psd = cv_scores(mk, Xdf)
        res_rows.append([mname, sname, "%.3f +/- %.3f" % (a, asd), "%.3f +/- %.3f" % (p, psd)])
sup = pd.DataFrame(res_rows, columns=["Model", "Features", "ROC_AUC (mean +/- sd)", "Avg_Precision (mean +/- sd)"])
print("\n" + "=" * 70 + "\nSUPERVISED MODELS: 5x5 repeated stratified CV\n" + "=" * 70)
print(sup.to_string(index=False))

# =====================================================================
# 6. FINAL MODEL, THRESHOLD CHOICE AND FLAGGED TRANSACTIONS
# =====================================================================
Xfinal = df[["LogAmount"]].to_numpy(float)
final = logreg()
oof = cross_val_predict(final, Xfinal, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        method="predict_proba")[:, 1]
print("\n" + "=" * 70 + "\nTHRESHOLD SELECTION (out-of-fold probabilities, never the test labels)\n" + "=" * 70)
COST_FN, COST_FP = 10.0, 1.0      # assumption: a missed fraud costs 10x an unnecessary review
rows = []
for t in np.arange(0.05, 1.0, 0.05):
    pred = (oof >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec = tp / (tp + fp) if tp + fp else np.nan
    rows.append([round(t, 2), tp, fp, fn, prec, tp / (tp + fn), COST_FN * fn + COST_FP * fp])
thr = pd.DataFrame(rows, columns=["Threshold", "TP", "FP", "FN", "Precision", "Recall", "Cost"])
print(thr.round(3).to_string(index=False))
best_t = float(thr.loc[thr["Cost"].idxmin(), "Threshold"])
print("\nThreshold with the lowest cost (FN cost %.0fx FP cost): %.2f" % (COST_FN / COST_FP, best_t))

pred = (oof >= best_t).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print("At threshold %.2f: TP=%d FP=%d FN=%d TN=%d | precision=%.3f recall=%.3f" % (
    best_t, tp, fp, fn, tn, tp / (tp + fp), tp / (tp + fn)))
missed = df[(pred == 0) & (y == 1)]["Amount"]
print("Missed fraud cases: %d, amounts %.0f to %.0f (median %.0f)" % (len(missed), missed.min(), missed.max(), missed.median()))
legit_hi = df.loc[df.Is_Fraud == 0, "Amount"].quantile(.99)
print("99th percentile of legitimate amounts: %.1f" % legit_hi)

prec, rec, th = precision_recall_curve(y, oof)
plt.figure(figsize=(7, 4.5))
plt.plot(th, prec[:-1], label="Precision"); plt.plot(th, rec[:-1], label="Recall")
plt.axvline(best_t, color="grey", ls="--", label="chosen threshold")
plt.xlabel("Threshold"); plt.title("Precision and recall vs threshold (out-of-fold)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/threshold.png", dpi=150); plt.close()

# Fit on all data, score every transaction, export the flagged list
final.fit(Xfinal, y)
coef = final[-1].coef_[0][0]
print("\nFinal model: log-odds change per 1 sd of log amount = %.2f" % coef)
df["FraudProbability"] = final.predict_proba(Xfinal)[:, 1]
flagged = df[df["FraudProbability"] >= best_t].sort_values("FraudProbability", ascending=False)
out = flagged[["Transaction_ID", "Date", "Time", "Amount", "Merchant", "Card_Number", "Transaction_Type",
               "RobustZ", "FraudProbability", "Is_Fraud"]].copy()
out[["RobustZ", "FraudProbability"]] = out[["RobustZ", "FraudProbability"]].round(3)
out.to_csv("flagged_transactions.csv", index=False)
print("Flagged %d transactions (saved to flagged_transactions.csv); %d are labelled fraud" % (
    len(flagged), flagged["Is_Fraud"].sum()))
print("Smallest flagged amount: %.2f" % flagged["Amount"].min())

# Compare the stage-1 rule: amount above the largest legitimate value
rule = (df["Amount"] > max_legit).astype(int)
tn, fp, fn, tp = confusion_matrix(y, rule).ravel()
print("\nSimple rule 'Amount > %.1f': TP=%d FP=%d FN=%d (precision %.2f, recall %.2f)" % (
    max_legit, tp, fp, fn, tp / (tp + fp), tp / (tp + fn)))