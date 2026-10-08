import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans']

#load data
df = pd.read_csv(
    "spiE_readmapping_supplemental.txt",
    sep="\t"
)

#filter out samples with reads < 1000000
df = df[df['Total_Reads'] >= 1000000]
df = df[df['Cohort'] != "HT"]
df = df.assign(coprostanol_present=df['Coprostanol_AUC'] > 0.1)
print("Dropped samples:")
print(df[df['Total_Reads'] < 1000000])
print(df[df['Cohort'] == "HT"])

#this is the y variable (what we want to predict)
print("selecting y var")
print(df[df['coprostanol_present'] == True])
y_true = df['coprostanol_present'].astype(int) #1 = positive for coprostanol, 0 = negative
print(y_true)

#this is the x variable (continuous variable we are using to predict y var)
cpm = [
    "CPM", 
]

#curve colors, legend tables
STYLE = {
    "CPM" : ("CPM", "#83b379")
}

#plot one ROC curve function
def add_roc(ax, y_true_series, score_series, label, colour, **plot_kw):
    mask = y_true_series.notna() & score_series.notna()
    y_cln, s_cln = y_true_series[mask].astype(int), score_series[mask]

    if len(np.unique(y_cln)) < 2:
        return 

    fpr, tpr, _ = roc_curve(y_cln, s_cln)
    auc = roc_auc_score(y_cln, s_cln)
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})",
            color=colour, lw=2, **plot_kw)

#draw panels
fig, ax = plt.subplots(
    1, 1, figsize=(6,6), sharex=True, sharey=True
)

#plot ROC curve for coprostanol
for col in cpm:
    lbl, col_hex = STYLE[col]
    add_roc(ax, y_true, df[col], lbl, col_hex)
ax.set_title("ROC: Coprostanol")
ax.set_xlabel("False-Positive Rate")
ax.set_ylabel("True-Positive Rate")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

#save figure
fig.tight_layout()
fig.savefig("spiE.svg", format="svg")
plt.show()
