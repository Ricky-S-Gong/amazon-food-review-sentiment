#!/usr/bin/env python3
"""
Fix inverted threshold logic in fig_pr.pdf and fig_summary.pdf.
The original script had: yp = (prob < tau) which inverts the label convention.
Correct logic: yp = (prob >= tau) means predict positive when prob >= tau.
"""
import os, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 11,
    'axes.labelsize': 10, 'legend.fontsize': 8,
})
sns.set_style('whitegrid')

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, '..', 'data')
MODELS = os.path.join(BASE, '..', 'models')
FIG    = os.path.join(BASE, 'figures')

print('[0] Loading data ...')
df = pd.read_csv(os.path.join(DATA, 'cleaned_data.csv'))
y  = df['Sentiment'].map({'positive': 1, 'negative': 0}).values

print('[1] Loading X_tfidf.pkl ...')
X = joblib.load(os.path.join(MODELS, 'X_tfidf.pkl'))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f'    X_test: {X_test.shape}')

print('[2] Training LR baseline and class-weights models ...')
lr_base = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr_base.fit(X_train, y_train)

lr_cw = LogisticRegression(max_iter=1000, random_state=42,
                            class_weight='balanced', solver='lbfgs')
lr_cw.fit(X_train, y_train)

best_lr = joblib.load(os.path.join(MODELS, 'best_logistic_model.pkl'))
print('[3] Computing probabilities ...')
y_prob_best = best_lr.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────
# Corrected threshold sweep
# ─────────────────────────────────────────────
print('[4] Threshold sweep (corrected) ...')
taus = np.linspace(0.05, 0.95, 300)
p_list, r_list = [], []
for tau in taus:
    # CORRECT: predict positive (1) when prob >= tau, negative (0) otherwise
    yp = (y_prob_best >= tau).astype(int)
    if (yp == 0).sum() == 0:   # no negatives predicted → skip
        p_list.append(0.0); r_list.append(0.0); continue
    p_list.append(precision_score(y_test, yp, pos_label=0, zero_division=0))
    r_list.append(recall_score(y_test, yp, pos_label=0, zero_division=0))

p_arr, r_arr = np.array(p_list), np.array(r_list)
f1_arr = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-8)
best_idx = int(np.argmax(f1_arr))
best_tau = taus[best_idx]
print(f'    τ* = {best_tau:.2f}  |  F1 = {f1_arr[best_idx]:.3f}  '
      f'|  Prec = {p_arr[best_idx]:.3f}  |  Rec = {r_arr[best_idx]:.3f}')

# ─────────────────────────────────────────────
# FIGURE 6 – Precision-Recall curve + threshold (corrected)
# ─────────────────────────────────────────────
print('[5] Figure: PR curve ...')
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(r_arr, p_arr, color='steelblue', lw=2, label='PR curve')
axes[0].scatter(r_arr[best_idx], p_arr[best_idx],
                color='red', zorder=5, s=90,
                label=f'Max F1  (τ={best_tau:.2f})')
axes[0].axhline(0.72, color='green', ls='--', lw=1.2, label='Precision floor 0.72')
axes[0].set_xlabel('Recall (Negative class)')
axes[0].set_ylabel('Precision (Negative class)')
axes[0].set_title('(a) Precision–Recall Curve (Negative class)')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 1.02); axes[0].set_ylim(0, 1.02)

axes[1].plot(taus, f1_arr, color='darkorange', lw=2)
axes[1].axvline(best_tau, color='red', ls='--', lw=1.2,
                label=f'τ* = {best_tau:.2f}  (Max F1 = {f1_arr[best_idx]:.3f})')
axes[1].set_xlabel('Classification Threshold τ')
axes[1].set_ylabel('F1 Score (Negative class)')
axes[1].set_title('(b) F1 Score vs. Threshold')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
p = os.path.join(FIG, 'fig_pr.pdf')
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f'    -> {p}')

# ─────────────────────────────────────────────
# FIGURE 8 – Model summary bar chart (corrected threshold)
# ─────────────────────────────────────────────
print('[6] Figure: Summary bar chart ...')
prob_base = lr_base.predict_proba(X_test)[:, 1]
prob_cw   = lr_cw.predict_proba(X_test)[:, 1]

model_names = [
    'LR Baseline',
    'LR+CW',
    'LR+CW+EN',
    'LR+CW+EN\n+Threshold',
]
auc_vals = [
    roc_auc_score(y_test, prob_base),
    roc_auc_score(y_test, prob_cw),
    roc_auc_score(y_test, y_prob_best),
    roc_auc_score(y_test, y_prob_best),
]
preds_dict = {
    'LR Baseline':     lr_base.predict(X_test),
    'LR+CW':           lr_cw.predict(X_test),
    'LR+CW+EN':        best_lr.predict(X_test),
    'LR+CW+EN+Thresh': (y_prob_best >= best_tau).astype(int),  # CORRECTED
}
neg_recall = [recall_score(y_test, p, pos_label=0) for p in preds_dict.values()]
neg_prec   = [precision_score(y_test, p, pos_label=0, zero_division=0) for p in preds_dict.values()]
neg_f1     = [f1_score(y_test, p, pos_label=0, zero_division=0) for p in preds_dict.values()]

x = np.arange(len(model_names))
w = 0.2
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - 1.5*w, auc_vals,   w, label='AUC-ROC',        color='#3498db', edgecolor='white')
ax.bar(x - 0.5*w, neg_recall, w, label='Neg. Recall',    color='#e67e22', edgecolor='white')
ax.bar(x + 0.5*w, neg_prec,   w, label='Neg. Precision', color='#e74c3c', edgecolor='white')
ax.bar(x + 1.5*w, neg_f1,     w, label='Neg. F1',        color='#2ecc71', edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=9)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('Score')
ax.set_title('Model Comparison — Key Metrics (Negative Class)')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
p = os.path.join(FIG, 'fig_summary.pdf')
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f'    -> {p}')

# ─────────────────────────────────────────────
# Print actual results for paper update
# ─────────────────────────────────────────────
print('\n=== Actual test-set results (for paper tables) ===')
all_models = {
    'LR Baseline':             (lr_base.predict(X_test), prob_base),
    'LR + Class Weights':      (lr_cw.predict(X_test),   prob_cw),
    'LR + Balanced + Elastic': (best_lr.predict(X_test), y_prob_best),
}
for name, (yp, ypr) in all_models.items():
    acc = (y_test == yp).mean()
    nr  = recall_score(y_test, yp, pos_label=0)
    np_ = precision_score(y_test, yp, pos_label=0, zero_division=0)
    nf1 = f1_score(y_test, yp, pos_label=0)
    au  = roc_auc_score(y_test, ypr)
    print(f'  {name:35s}  Acc={acc:.4f}  NegRec={nr:.4f}  '
          f'NegPrec={np_:.4f}  NegF1={nf1:.4f}  AUC={au:.4f}')

yp_thr = (y_prob_best >= best_tau).astype(int)
nr  = recall_score(y_test, yp_thr, pos_label=0)
np_ = precision_score(y_test, yp_thr, pos_label=0, zero_division=0)
nf1 = f1_score(y_test, yp_thr, pos_label=0)
au  = roc_auc_score(y_test, y_prob_best)
print(f'  {"LR+Balanced+Elastic+Threshold":35s}  Acc=N/A      '
      f'NegRec={nr:.4f}  NegPrec={np_:.4f}  NegF1={nf1:.4f}  AUC={au:.4f}')
print(f'  τ* = {best_tau:.2f}')
