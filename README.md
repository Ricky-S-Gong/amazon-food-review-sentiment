# Amazon Fine Food Review — Sentiment Analysis

> **NLP Binary Classification | Logistic Regression | Random Forest | Ensemble Learning**

---

## English Version

### Project Overview

This project performs **sentiment analysis** on the Amazon Fine Food Reviews dataset. The goal is to build a machine learning pipeline that predicts whether a customer review is **positive** or **negative** based solely on the text content. The project covers the full data science workflow: text preprocessing, feature engineering, class imbalance handling, model training and evaluation, and ensemble methods.

This is a classic NLP classification problem and a common interview topic at major tech companies. The insights derived from sentiment analysis can inform product improvement, vendor selection, and customer experience optimization.

---

### Dataset

| Property | Value |
|---|---|
| Source | [Amazon Fine Food Reviews (Kaggle)](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) |
| Total Records | 568,454 reviews |
| Features | Id, ProductId, UserId, ProfileName, Score, Summary, Text, Time, etc. |
| Label | Positive (Score > 3) / Negative (Score < 3); Score = 3 excluded |
| Class Distribution | Positive: 78.1% / Negative: 21.9% (imbalanced) |

> **Note:** Due to file size constraints, raw data files are not stored in this repository. Please download `Reviews.csv` from the Kaggle link above and place it in the `data/` directory before running the notebooks.

---

### Project Structure

```
Amazon-Food-Review-Sentiment/
├── notebooks/
│   ├── amazon_food_review.ipynb        # Course template notebook
│   └── ztgg_project_amazon_solution.ipynb  # Full solution notebook
├── docs/
│   ├── (Amazon)Project lecture.pdf     # Project lecture slides
│   ├── Amazon Instruction.docx         # Project instructions
│   └── Amazon statement.docx           # Project statement
├── paper/
│   └── amazon_sentiment_paper.tex      # LaTeX paper report
├── data/                               # (gitignored) Place raw data here
├── models/                             # (gitignored) Saved model files
├── .gitignore
└── README.md
```

---

### Methodology

#### 1. Data Preprocessing

- **Label Engineering**: Map numeric scores to binary sentiment (positive/negative); ignore neutral scores (= 3)
- **Text Cleaning**: Lowercasing, punctuation removal, tokenization
- **Stemming**: Porter Stemmer / Snowball Stemmer to reduce vocabulary size
- **Stop Words**: Intentionally retained negation words (e.g., "not", "won't") since they carry strong sentiment signals
- **Visualization**: Word clouds for high-score vs low-score reviews

#### 2. Feature Engineering

| Method | Description | Shape |
|---|---|---|
| Uni-gram BOW | Binary term presence, min_df=5 | (454763, 8208) |
| Bi-gram BOW | Unigram + bigram, min_df=5 | (454763, 46995) |
| TF-IDF | 1-2 gram, max_features=10000 | (568454, 10000) |
| TruncatedSVD | Dimensionality reduction to 1000 components | (N, 1000) |

#### 3. Handling Class Imbalance

Three strategies were explored and compared:

| Strategy | Negative Recall | Negative Precision | AUC |
|---|---|---|---|
| Class Weight (`balanced`) | 0.872 | 0.661 | 0.944 |
| SMOTE Oversampling | 0.818 | 0.613 | 0.916 |
| Random Undersampling | 0.873 | 0.648 | 0.941 |

> Business rationale: **Recall is prioritized for the negative class** — missing a truly negative review (false negative) is more costly than a false alarm.

#### 4. Models

**Part 1 — Logistic Regression**
- Baseline accuracy: 90.2%; negative recall: 69.1%
- With `class_weight='balanced'`: negative recall rises to 87.2%
- Best model: Balanced + ElasticNet regularization (`l1_ratio=0.5`, GridSearchCV 3-fold CV)
  - Negative Recall: **0.875** | AUC: **0.945**

**Part 2 — Random Forest**
- GridSearchCV: n_estimators=500, max_depth=20, max_features='sqrt'
- Applied on both full TF-IDF and SVD-reduced feature spaces
- With `class_weight='balanced'` + SVD: precision improves but recall decreases

**Part 3 — Soft Voting Ensemble (Best Model)**
- Combines Logistic Regression (high recall) + Random Forest (high precision)
- Soft voting averages predicted probabilities from both classifiers
- **Final Results**: Negative Precision: 0.716 | Negative Recall: 0.844 | **AUC: 0.9506**

#### 5. Feature Importance Analysis

**Top positive sentiment keywords (Logistic Regression coefficients):**
`four star`, `perfect`, `highly recommend`, `delicious`, `great`, `love`, `best`

**Top negative sentiment keywords:**
`three star`, `worst`, `won't buy`, `disappoint`, `two star`, `terrible`, `horrible`

---

### Key Results Summary

| Model | Accuracy | Neg. Recall | Neg. Precision | Neg. F1 | AUC |
|---|---|---|---|---|---|
| LR Baseline | 90.2% | 0.691 | 0.828 | 0.753 | 0.943 |
| LR + Balanced | 87.5% | 0.872 | 0.661 | 0.752 | 0.944 |
| LR + Balanced + ElasticNet | 87.7% | 0.875 | 0.665 | 0.755 | 0.945 |
| RF + Balanced + SVD | ~83% | 0.603 | 0.876 | 0.714 | 0.902 |
| **Soft Voting (Final)** | **~88%** | **0.844** | **0.716** | **0.775** | **0.951** |

---

### How to Run

**Prerequisites:**
```bash
pip install pandas numpy nltk scikit-learn imbalanced-learn matplotlib seaborn wordcloud joblib tqdm
```

**Steps:**
1. Download `Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and place in `data/`
2. Open `notebooks/ztgg_project_amazon_solution.ipynb`
3. Run cells in order — data cleaning, feature engineering, model training, evaluation

---

### Interview Talking Points

> "I built a sentiment classification model on Amazon Fine Food Reviews (568K records) to predict positive/negative user sentiment from review text. I applied TF-IDF vectorization with bigrams, addressed a significant class imbalance (78/22 split) using class weighting and SMOTE, and trained Logistic Regression and Random Forest classifiers. The final ensemble (Soft Voting) achieved AUC 0.951, with particular emphasis on maximizing negative-class recall to ensure dissatisfied customers are reliably detected."

---

### Tech Stack

- **Language**: Python 3
- **NLP**: NLTK (stemming, stop words), scikit-learn (TF-IDF, CountVectorizer)
- **ML**: scikit-learn (LogisticRegression, RandomForestClassifier, VotingClassifier, GridSearchCV, TruncatedSVD)
- **Imbalance Handling**: imbalanced-learn (SMOTE, RandomUnderSampler)
- **Visualization**: matplotlib, seaborn, WordCloud
- **Persistence**: joblib, pickle

---

---

## 中文版本

### 项目简介

本项目基于亚马逊精选食品评论数据集，构建**情感分析**机器学习流水线，通过用户评论的文本内容预测该评论为**正面**还是**负面**情感。项目涵盖完整的数据科学工作流：文本预处理、特征工程、标签不平衡处理、模型训练与评估、以及集成学习方法。

情感分析（即观点挖掘 opinion mining）是当前各大公司的面试热点，分析结果可服务于产品改进、供应商筛选和用户体验优化等业务场景。

---

### 数据集说明

| 属性 | 说明 |
|---|---|
| 来源 | Amazon Fine Food Reviews（Kaggle） |
| 总量 | 568,454 条评论 |
| 特征 | 用户ID、商品ID、评分、摘要、正文等 |
| 标签构建 | 正类：Score > 3；负类：Score < 3；Score = 3 忽略 |
| 类别分布 | 正类 78.1% / 负类 21.9%（显著不平衡） |

> **注意**：由于文件体积过大，原始数据未纳入本仓库。请从 Kaggle 下载 `Reviews.csv` 并放置于 `data/` 目录后再运行 notebook。

---

### 项目结构

```
Amazon-Food-Review-Sentiment/
├── notebooks/           # Jupyter notebooks
├── docs/                # 项目讲义、说明文档
├── paper/               # LaTeX 学术报告
├── data/                # （已 gitignore）原始/清洗后数据
├── models/              # （已 gitignore）保存的模型文件
├── .gitignore
└── README.md
```

---

### 技术方法

#### 数据预处理
- 标签二值化（正/负），忽略中性评分（=3）
- 文本清洗：小写化、去标点、分词
- 词干提取（Porter/Snowball Stemmer）
- 保留否定词（如 "not"）因其具有强烈情感信号
- 词云可视化高/低分评论关键词

#### 特征工程
- Uni-gram BOW、Bi-gram BOW、TF-IDF（1-2 gram）
- TruncatedSVD 将特征降维至 1,000 维

#### 标签不平衡处理
| 策略 | 负类 Recall | 负类 Precision | AUC |
|---|---|---|---|
| 类别权重 (balanced) | 0.872 | 0.661 | 0.944 |
| SMOTE 上采样 | 0.818 | 0.613 | 0.916 |
| 随机下采样 | 0.873 | 0.648 | 0.941 |

> 业务逻辑：**优先保证负类 Recall** —— 漏判负面评价（false negative）的业务代价远高于误报。

#### 建模流程

**Part 1 — 逻辑回归**
- 基线模型准确率 90.2%，负类 Recall 仅 69.1%
- 引入类别权重后负类 Recall 提升至 87.2%
- 最优：Balanced + ElasticNet 正则（l1_ratio=0.5，3折交叉验证调参）

**Part 2 — 随机森林**
- GridSearchCV 调参：n_estimators=500, max_depth=20
- 降维后负类 Precision 更高，整体 AUC 0.90+

**Part 3 — 软投票集成（最终模型）**
- 融合逻辑回归（高 Recall）+ 随机森林（高 Precision）
- 通过概率平均实现两者互补
- **最终结果**：负类 Precision 0.716，Recall 0.844，**AUC 0.9506**

---

### 最终结果对比

| 模型 | Accuracy | 负类 Recall | 负类 Precision | 负类 F1 | AUC |
|---|---|---|---|---|---|
| LR 基线 | 90.2% | 0.691 | 0.828 | 0.753 | 0.943 |
| LR + 类别权重 | 87.5% | 0.872 | 0.661 | 0.752 | 0.944 |
| LR + 权重 + ElasticNet | 87.7% | 0.875 | 0.665 | 0.755 | 0.945 |
| RF + 权重 + SVD | ~83% | 0.603 | 0.876 | 0.714 | 0.902 |
| **软投票集成（最终）** | **~88%** | **0.844** | **0.716** | **0.775** | **0.951** |

---

### 运行方法

```bash
pip install pandas numpy nltk scikit-learn imbalanced-learn matplotlib seaborn wordcloud joblib tqdm
```

1. 从 Kaggle 下载 `Reviews.csv` 放入 `data/` 目录
2. 打开 `notebooks/ztgg_project_amazon_solution.ipynb`
3. 按顺序运行所有 Cell

---

### 技术栈

- **语言**：Python 3
- **NLP**：NLTK、scikit-learn（TF-IDF、CountVectorizer）
- **机器学习**：scikit-learn（逻辑回归、随机森林、VotingClassifier、GridSearchCV、TruncatedSVD）
- **不平衡处理**：imbalanced-learn（SMOTE、RandomUnderSampler）
- **可视化**：matplotlib、seaborn、WordCloud
- **模型持久化**：joblib、pickle
