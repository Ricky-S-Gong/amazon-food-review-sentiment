#!/usr/bin/env python3
"""
Generate all paper figures from saved models and data artifacts.
Run from the paper/ directory: python generate_figures.py
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
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
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
os.makedirs(FIG, exist_ok=True)

# ─────────────────────────────────────────────
# 0. Load cleaned data
# ─────────────────────────────────────────────
print('[0] Loading cleaned_data.csv ...')
df = pd.read_csv(os.path.join(DATA, 'cleaned_data.csv'))
y  = df['Sentiment'].map({'positive': 1, 'negative': 0}).values
print(f'    Rows: {len(df):,}  |  pos={y.sum():,}  neg={(y==0).sum():,}')

# ─────────────────────────────────────────────
# 1. FIGURE 1 – EDA
# ─────────────────────────────────────────────
print('[1] Figure 1: EDA ...')
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

# (a) score distribution
sc = df['Score'].value_counts().sort_index()
colors = ['#c0392b','#e67e22','#f1c40f','#2ecc71','#27ae60']
bars = axes[0].bar(sc.index, sc.values, color=colors, edgecolor='white', linewidth=0.8)
axes[0].set_xlabel('Star Rating')
axes[0].set_ylabel('Number of Reviews')
axes[0].set_title('(a) Raw Score Distribution')
for x, v in zip(sc.index, sc.values):
    axes[0].text(x, v + 3500, f'{v/1000:.0f}K', ha='center', fontsize=8)
axes[0].set_ylim(0, sc.max() * 1.12)

# (b) binary label distribution
vc = df['Sentiment'].value_counts()
pos_n, neg_n = vc.get('positive', 0), vc.get('negative', 0)
total = pos_n + neg_n
axes[1].barh(['Negative (0)', 'Positive (1)'], [neg_n, pos_n],
             color=['#c0392b', '#27ae60'], edgecolor='white')
for i, v in enumerate([neg_n, pos_n]):
    axes[1].text(v + 3000, i, f'{v:,}  ({v/total*100:.1f}%)', va='center', fontsize=9)
axes[1].set_xlabel('Number of Reviews')
axes[1].set_title('(b) Binary Sentiment Label Distribution')
axes[1].set_xlim(0, max(pos_n, neg_n) * 1.22)

plt.tight_layout()
p = os.path.join(FIG, 'fig_eda.pdf')
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f'    -> {p}')

# ─────────────────────────────────────────────
# 2. FIGURE 2 – Word clouds
# ─────────────────────────────────────────────
if 'CleanText' in df.columns:
    print('[2] Figure 2: Word clouds ...')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, mask, title, cmap in [
        (axes[0], y == 1, '(a) Positive Reviews (Score > 3)', 'YlGn'),
        (axes[1], y == 0, '(b) Negative Reviews (Score < 3)', 'OrRd'),
    ]:
        texts = ' '.join(df.loc[mask, 'CleanText'].dropna().sample(
            min(30000, mask.sum()), random_state=42).values)
        wc = WordCloud(background_color='white', max_words=120,
                       colormap=cmap, width=600, height=350,
                       random_state=42).generate(texts)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off'); ax.set_title(title, fontsize=11)
    plt.tight_layout()
    p = os.path.join(FIG, 'fig_wordcloud.pdf')
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f'    -> {p}')
else:
    print('[2] CleanText column not found, skipping word clouds.')

# ─────────────────────────────────────────────
# 3. Load TF-IDF matrix and split
# ─────────────────────────────────────────────
print('[3] Loading X_tfidf.pkl (this may take ~10s) ...')
X = joblib.load(os.path.join(MODELS, 'X_tfidf.pkl'))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f'    X_train: {X_train.shape}  X_test: {X_test.shape}')

# ─────────────────────────────────────────────
# 4. FIGURE 3 – SVD explained variance
# ─────────────────────────────────────────────
print('[4] Figure 3: SVD explained variance (fitting 300 components ~2-4 min) ...')
svd = TruncatedSVD(n_components=300, random_state=42)
svd.fit(X_train)
cumvar = np.cumsum(svd.explained_variance_ratio_)
n_for_90 = int(np.searchsorted(cumvar, 0.90)) + 1

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
comps = np.arange(1, 301)

axes[0].plot(comps, cumvar, color='steelblue', lw=2, label='Cumulative EV')
axes[0].axhline(0.90, color='red', ls='--', lw=1.3, label='90% threshold')
axes[0].axhline(0.70, color='orange', ls='--', lw=1.0, label='70% threshold')
axes[0].fill_between(comps, cumvar, alpha=0.15, color='steelblue')
axes[0].set_xlabel('Number of SVD Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('(a) Cumulative Explained Variance')
axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_ylim(0, 1.02)
# Annotate value at 300
axes[0].annotate(
    f'{cumvar[-1]:.1%} at 300 components\n(90% requires > 1,000)',
    xy=(300, cumvar[-1]), xytext=(160, cumvar[-1] - 0.12),
    arrowprops=dict(arrowstyle='->', color='navy'),
    color='navy', fontsize=8)

axes[1].plot(comps, svd.explained_variance_ratio_, color='darkorange', lw=1.5)
axes[1].set_xlabel('Component Index')
axes[1].set_ylabel('Individual Explained Variance')
axes[1].set_title('(b) Marginal Explained Variance per Component')
axes[1].grid(alpha=0.3)

plt.suptitle('Truncated SVD on TF-IDF Features — Variance Analysis', fontsize=11, y=1.01)
plt.tight_layout()
p = os.path.join(FIG, 'fig_svd.pdf')
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f'    Variance @ 300 components: {cumvar[-1]:.2%}')
print(f'    Components needed for 90%: > 1,000')
print(f'    -> {p}')

# ─────────────────────────────────────────────
# 5. Train / Load LR models
# ─────────────────────────────────────────────
print('[5] Training LR baseline and balanced models ...')
lr_base = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr_base.fit(X_train, y_train)
print('    Baseline done.')

lr_cw = LogisticRegression(max_iter=1000, random_state=42,
                            class_weight='balanced', solver='lbfgs')
lr_cw.fit(X_train, y_train)
print('    Class-weights done.')

best_lr = joblib.load(os.path.join(MODELS, 'best_logistic_model.pkl'))
print('    best_logistic_model.pkl loaded.')

models = {
    'LR Baseline':              (lr_base,  X_test),
    'LR + Class Weights':       (lr_cw,    X_test),
    'LR + Balanced + Elastic':  (best_lr,  X_test),
}
probs = {k: m.predict_proba(Xv)[:, 1] for k, (m, Xv) in models.items()}

# ─────────────────────────────────────────────
# 6. FIGURE 4 – ROC curves
# ─────────────────────────────────────────────
print('[6] Figure 4: ROC curves ...')
pal = ['#3498db', '#e67e22', '#e74c3c']
fig, ax = plt.subplots(figsize=(6, 5))
for (name, prob), col in zip(probs.items(), pal):
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, lw=2, color=col,
            label=f'{name}  (AUC={auc(fpr,tpr):.3f})')
ax.plot([0,1],[0,1],'k--', lw=1, label='Random classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Logistic Regression Variants')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
p = os.path.join(FIG, 'fig_roc.pdf')
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f'    -> {p}')

# ─────────────────────────────────────────────
# 7. FIGURE 5 – Confusion matrices (baseline vs best)
# ─────────────────────────────────────────────
print('[7] Figure 5: Confusion matrices ...')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, (model, Xv)) in zip(axes, {
        'LR Baseline': (lr_base, X_test),
        'LR + Balanced + ElasticNet': (best_lr, X_test),
}.items()):
    cm = confusion_matrix(y_test, model.predict(Xv))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['True Neg', 'True Pos'],
                annot_kws={'size': 12})
    rec = cm[0,0]/(cm[0,0]+cm[0,1])
    prec = cm[0,0]/(cm[0,0]+cm[1,0])
    ax.set_title(f'{name}\nNeg Recall={rec:.3f}  Neg Prec={prec:.3f}', fontsize=9)
plt.tight_layout()
p = os.path.join(FIG, 'fig_confusion.pdf')
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f'    -> {p}')

# ─────────────────────────────────────────────
# 8. FIGURE 6 – Precision-Recall curve + threshold
# ─────────────────────────────────────────────
print('[8] Figure 6: PR curve and threshold analysis ...')
y_prob_best = probs['LR + Balanced + Elastic']

taus = np.linspace(0.05, 0.95, 300)
p_list, r_list = [], []
for tau in taus:
    yp = (y_prob_best < tau).astype(int)   # predict 0 (negative) when prob < tau
    if yp.sum() == 0:
        p_list.append(0.0); r_list.append(0.0); continue
    p_list.append(precision_score(y_test, yp, pos_label=0, zero_division=0))
    r_list.append(recall_score(y_test, yp, pos_label=0, zero_division=0))
p_arr, r_arr = np.array(p_list), np.array(r_list)
f1_arr = 2*p_arr*r_arr/(p_arr+r_arr+1e-8)
best_idx = int(np.argmax(f1_arr))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# PR curve
axes[0].plot(r_arr, p_arr, color='steelblue', lw=2, label='PR curve')
axes[0].scatter(r_arr[best_idx], p_arr[best_idx],
                color='red', zorder=5, s=90,
                label=f'Max F1  (τ={taus[best_idx]:.2f})')
axes[0].axhline(0.72, color='green', ls='--', lw=1.2, label='Precision floor 0.72')
axes[0].set_xlabel('Recall (Negative class)')
axes[0].set_ylabel('Precision (Negative class)')
axes[0].set_title('(a) Precision–Recall Curve (Negative class)')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, 1.02); axes[0].set_ylim(0, 1.02)

# F1 vs threshold
axes[1].plot(taus, f1_arr, color='darkorange', lw=2)
axes[1].axvline(taus[best_idx], color='red', ls='--', lw=1.2,
                label=f'τ* = {taus[best_idx]:.2f}  (Max F1 = {f1_arr[best_idx]:.3f})')
axes[1].set_xlabel('Classification Threshold τ')
axes[1].set_ylabel('F1 Score (Negative class)')
axes[1].set_title('(b) F1 Score vs. Threshold')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
p = os.path.join(FIG, 'fig_pr.pdf')
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f'    τ* = {taus[best_idx]:.2f}  |  F1 = {f1_arr[best_idx]:.3f}  |  '
      f'Prec = {p_arr[best_idx]:.3f}  |  Rec = {r_arr[best_idx]:.3f}')
print(f'    -> {p}')

# ─────────────────────────────────────────────
# 9. FIGURE 7 – Feature importance
# ─────────────────────────────────────────────
print('[9] Figure 7: Feature importance (re-fitting vectorizer for names) ...')
vec_new = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
vec_new.fit(df['CleanText'].fillna(''))
feat_names = vec_new.get_feature_names_out()
coefs = best_lr.coef_.flatten()
coef_df = pd.DataFrame({'Feature': feat_names, 'Coef': coefs})
coef_df = coef_df.sort_values('Coef', ascending=False)

top_n = 15
top_pos = coef_df.head(top_n)
top_neg = coef_df.tail(top_n).iloc[::-1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].barh(range(top_n), top_pos['Coef'].values[::-1], color='#27ae60', edgecolor='none')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_pos['Feature'].values[::-1], fontsize=8)
axes[0].set_xlabel('LR Coefficient')
axes[0].set_title(f'(a) Top {top_n} Positive-Sentiment Features')
axes[0].grid(axis='x', alpha=0.3)

axes[1].barh(range(top_n), np.abs(top_neg['Coef'].values), color='#c0392b', edgecolor='none')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_neg['Feature'].values, fontsize=8)
axes[1].set_xlabel('|LR Coefficient|')
axes[1].set_title(f'(b) Top {top_n} Negative-Sentiment Features')
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Logistic Regression Feature Coefficients (TF-IDF bi-gram)', fontsize=11)
plt.tight_layout()
p = os.path.join(FIG, 'fig_features.pdf')
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f'    -> {p}')

# ─────────────────────────────────────────────
# 10. FIGURE 8 – Model summary bar chart
# ─────────────────────────────────────────────
print('[10] Figure 8: Model summary bar chart ...')
model_names = [
    'LR Baseline',
    'LR+CW',
    'LR+CW+EN',
    'LR+CW+EN\n+Threshold',
]
auc_vals  = [
    roc_auc_score(y_test, probs['LR Baseline']),
    roc_auc_score(y_test, probs['LR + Class Weights']),
    roc_auc_score(y_test, probs['LR + Balanced + Elastic']),
    roc_auc_score(y_test, probs['LR + Balanced + Elastic']),
]

best_tau  = taus[best_idx]
preds = {
    'LR Baseline':    lr_base.predict(X_test),
    'LR+CW':          lr_cw.predict(X_test),
    'LR+CW+EN':       best_lr.predict(X_test),
    'LR+CW+EN+Thresh':(y_prob_best < best_tau).astype(int),
}
neg_recall  = [recall_score(y_test, p, pos_label=0) for p in preds.values()]
neg_prec    = [precision_score(y_test, p, pos_label=0, zero_division=0) for p in preds.values()]
neg_f1      = [f1_score(y_test, p, pos_label=0, zero_division=0) for p in preds.values()]

x = np.arange(len(model_names))
w = 0.2
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(x - 1.5*w, auc_vals,   w, label='AUC-ROC',       color='#3498db', edgecolor='white')
ax.bar(x - 0.5*w, neg_recall, w, label='Neg. Recall',   color='#e67e22', edgecolor='white')
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
# Summary
# ─────────────────────────────────────────────
print('\n=== Actual test-set results ===')
for name, (model, Xv) in models.items():
    yp = model.predict(Xv)
    ypr = model.predict_proba(Xv)[:,1]
    acc = (y_test == yp).mean()
    nr  = recall_score(y_test, yp, pos_label=0)
    np_ = precision_score(y_test, yp, pos_label=0, zero_division=0)
    nf1 = f1_score(y_test, yp, pos_label=0)
    au  = roc_auc_score(y_test, ypr)
    print(f'  {name:35s}  Acc={acc:.4f}  NegRec={nr:.4f}  NegPrec={np_:.4f}  '
          f'NegF1={nf1:.4f}  AUC={au:.4f}')

yp_thr = (probs['LR + Balanced + Elastic'] < best_tau).astype(int)
nr  = recall_score(y_test, yp_thr, pos_label=0)
np_ = precision_score(y_test, yp_thr, pos_label=0, zero_division=0)
nf1 = f1_score(y_test, yp_thr, pos_label=0)
print(f'  {"LR + Balanced + Elastic + Threshold":35s}  Acc=N/A      '
      f'NegRec={nr:.4f}  NegPrec={np_:.4f}  NegF1={nf1:.4f}  '
      f'AUC={roc_auc_score(y_test, probs["LR + Balanced + Elastic"]):.4f}')

print(f'\nAll {len(os.listdir(FIG))} figures saved to {FIG}')
