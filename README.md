# Amazon Fine Food Reviews — Sentiment Analysis

> **NLP Binary Classification | Logistic Regression | Random Forest | Ensemble Learning**

---

## English

### Overview

This project builds a binary sentiment classifier on the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset (568K reviews). The pipeline covers the full NLP workflow: text preprocessing, TF-IDF feature engineering, class imbalance handling, model training and evaluation, threshold tuning, and soft-voting ensemble. A companion academic paper (`paper/amazon_sentiment_paper.pdf`) documents the methodology and findings in depth.

---

### Repository Structure

```
amazon-food-review-sentiment/
├── notebooks/
│   └── amazon_food_review_final.ipynb   # Main analysis notebook
├── data/
│   └── cleaned_data.xlsx                # Preprocessed dataset (derived from Kaggle source)
├── models/
│   └── best_logistic_model.pkl          # Best trained model (LR + Balanced + ElasticNet)
├── paper/
│   ├── amazon_sentiment_paper.tex       # LaTeX source
│   ├── amazon_sentiment_paper.pdf       # Compiled report
│   ├── figures/                         # 8 result figures
│   ├── generate_figures.py
│   ├── fix_threshold_figures.py
│   └── update_cells.py
├── .gitignore
└── README.md
```

> **Raw data not included.** Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place it in `data/` to re-run the full pipeline from scratch. `cleaned_data.xlsx` is provided for convenience.

---

### Dataset

| Property | Value |
|---|---|
| Source | Amazon Fine Food Reviews (Kaggle / SNAP) |
| Total Records | 568,454 reviews |
| Label | Positive (Score > 3) / Negative (Score < 3); Score = 3 excluded |
| Class Distribution | Positive 78.1% / Negative 21.9% (imbalanced) |

---

### Methodology

#### 1. Text Preprocessing
- Binary label encoding; neutral scores (= 3) excluded
- Lowercasing, punctuation removal, tokenization
- Snowball Stemmer for vocabulary reduction
- Negation words (e.g., "not", "won't") intentionally retained — strong sentiment signal

#### 2. Feature Engineering

| Method | Description |
|---|---|
| TF-IDF | 1–2 gram, `max_features=10,000` |
| TruncatedSVD | Dimensionality reduction (for SMOTE and RF) |

#### 3. Class Imbalance Strategies

| Strategy | Neg. Recall | Neg. Precision | AUC |
|---|---|---|---|
| Class Weight (`balanced`) | 0.871 | 0.665 | 0.944 |
| SMOTE (SVD-reduced) | 0.818 | 0.613 | 0.916 |
| Random Undersampling | 0.873 | 0.648 | 0.941 |

> **Business rationale:** Recall is prioritized for the negative class — missing a dissatisfied customer is costlier than a false alarm.

#### 4. Models

**Logistic Regression**
- Baseline: accuracy 90.0%, negative recall 69.1%
- `class_weight='balanced'` raises negative recall to 87.1%
- ElasticNet regularization (`l1_ratio=0.5`, 3-fold CV) further improves AUC to 0.951
- Threshold tuning at τ* = 0.37 yields best negative-class F₁ = **0.784**

**Random Forest**
- Trained on SVD-reduced features
- High precision (0.876) but lower recall (0.603) — complementary to LR

**Soft Voting Ensemble (final model)**
- Combines LR (high recall) + RF (high precision) via probability averaging
- Best overall balance of recall and AUC

---

### Results

| Model | Accuracy | Neg. Recall | Neg. Precision | Neg. F₁ | AUC |
|---|---|---|---|---|---|
| LR Baseline | 0.900 | 0.691 | 0.827 | 0.753 | 0.942 |
| LR + Class Weights | 0.875 | 0.871 | 0.665 | 0.754 | 0.944 |
| LR + Balanced + ElasticNet | 0.882 | 0.886 | 0.677 | 0.768 | 0.951 |
| LR + Balanced + EN + Threshold (τ*=0.37) | — | 0.821 | 0.750 | **0.784** | 0.951 |
| RF + Class Weights (SVD) | 0.870 | 0.603 | 0.876 | 0.714 | 0.902 |
| RF + SMOTE (SVD) | 0.890 | 0.670 | 0.790 | 0.730 | 0.937 |
| **Soft Voting (LR + RF)** | **0.882** | **0.844** | 0.716 | 0.775 | **0.951** |

---

### How to Run

**Install dependencies:**
```bash
pip install pandas numpy nltk scikit-learn imbalanced-learn matplotlib seaborn wordcloud joblib tqdm openpyxl
```

**Steps:**
1. (Optional) Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place in `data/` to run from raw data
2. Open `notebooks/amazon_food_review_final.ipynb`
3. Run cells in order

---

### Tech Stack

| Category | Libraries |
|---|---|
| NLP | NLTK, scikit-learn (TF-IDF, CountVectorizer) |
| ML | scikit-learn (LogisticRegression, RandomForestClassifier, VotingClassifier, TruncatedSVD, GridSearchCV) |
| Imbalance | imbalanced-learn (SMOTE, RandomUnderSampler) |
| Visualization | matplotlib, seaborn, WordCloud |
| Reporting | LaTeX (paper), joblib (model persistence) |

---

---

## 中文版

### 项目简介

本项目基于亚马逊精选食品评论数据集（568K 条），构建二分类情感分析流水线，通过评论文本预测用户情感（正面/负面）。完整覆盖 NLP 工作流：文本预处理、TF-IDF 特征工程、标签不平衡处理、模型训练与评估、阈值调优与集成学习。配套学术报告详见 `paper/amazon_sentiment_paper.pdf`。

---

### 项目结构

```
amazon-food-review-sentiment/
├── notebooks/
│   └── amazon_food_review_final.ipynb   # 主分析 notebook
├── data/
│   └── cleaned_data.xlsx                # 预处理后的数据集（来源于 Kaggle）
├── models/
│   └── best_logistic_model.pkl          # 最优模型（LR + Balanced + ElasticNet）
├── paper/
│   ├── amazon_sentiment_paper.tex       # LaTeX 源文件
│   ├── amazon_sentiment_paper.pdf       # 编译后报告
│   └── figures/                         # 8 张结果图
├── .gitignore
└── README.md
```

> **原始数据未包含。** 如需从头运行完整流水线，请从 [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) 下载 `Reviews.csv` 放入 `data/` 目录。`cleaned_data.xlsx` 已提供供直接使用。

---

### 数据集

| 属性 | 说明 |
|---|---|
| 来源 | Amazon Fine Food Reviews（Kaggle / SNAP） |
| 总量 | 568,454 条评论 |
| 标签 | 正类 Score > 3，负类 Score < 3，Score = 3 排除 |
| 类别分布 | 正类 78.1% / 负类 21.9%（显著不平衡） |

---

### 技术方法

#### 文本预处理
- 标签二值化，排除中性评分（=3）
- 小写化、去标点、分词
- Snowball Stemmer 词干提取
- 保留否定词（如 "not"、"won't"）——具有强烈情感信号

#### 特征工程
- TF-IDF（1–2 gram，max_features=10,000）
- TruncatedSVD 降维（用于 SMOTE 和随机森林）

#### 标签不平衡处理
| 策略 | 负类 Recall | 负类 Precision | AUC |
|---|---|---|---|
| 类别权重 (balanced) | 0.871 | 0.665 | 0.944 |
| SMOTE 上采样 | 0.818 | 0.613 | 0.916 |
| 随机下采样 | 0.873 | 0.648 | 0.941 |

> **业务逻辑：** 优先保证负类 Recall —— 漏判负面评价的代价远高于误报。

#### 建模流程

**逻辑回归**
- 基线：准确率 90.0%，负类 Recall 仅 69.1%
- 引入类别权重后 Recall 提升至 87.1%
- ElasticNet 正则（3折CV）进一步提升 AUC 至 0.951
- 阈值调优 τ* = 0.37，最优负类 F₁ = **0.784**

**随机森林**
- 在 SVD 降维特征上训练
- 负类 Precision 高（0.876），Recall 较低（0.603）——与 LR 互补

**软投票集成（最终模型）**
- 融合 LR（高 Recall）+ RF（高 Precision），取概率均值
- 综合表现最优

---

### 最终结果

| 模型 | Accuracy | 负类 Recall | 负类 Precision | 负类 F₁ | AUC |
|---|---|---|---|---|---|
| LR 基线 | 0.900 | 0.691 | 0.827 | 0.753 | 0.942 |
| LR + 类别权重 | 0.875 | 0.871 | 0.665 | 0.754 | 0.944 |
| LR + 权重 + ElasticNet | 0.882 | 0.886 | 0.677 | 0.768 | 0.951 |
| LR + 权重 + EN + 阈值(τ*=0.37) | — | 0.821 | 0.750 | **0.784** | 0.951 |
| RF + 类别权重(SVD) | 0.870 | 0.603 | 0.876 | 0.714 | 0.902 |
| RF + SMOTE(SVD) | 0.890 | 0.670 | 0.790 | 0.730 | 0.937 |
| **软投票集成（最终）** | **0.882** | **0.844** | 0.716 | 0.775 | **0.951** |

---

### 运行方法

```bash
pip install pandas numpy nltk scikit-learn imbalanced-learn matplotlib seaborn wordcloud joblib tqdm openpyxl
```

1. （可选）从 [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) 下载 `Reviews.csv` 放入 `data/`
2. 打开 `notebooks/amazon_food_review_final.ipynb`
3. 按顺序运行所有 Cell

---

### 技术栈

| 类别 | 库 |
|---|---|
| NLP | NLTK、scikit-learn（TF-IDF、CountVectorizer） |
| 机器学习 | scikit-learn（逻辑回归、随机森林、VotingClassifier、TruncatedSVD、GridSearchCV） |
| 不平衡处理 | imbalanced-learn（SMOTE、RandomUnderSampler） |
| 可视化 | matplotlib、seaborn、WordCloud |
| 报告 | LaTeX（学术报告）、joblib（模型持久化） |
