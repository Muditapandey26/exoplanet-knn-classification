# Exoplanet Classification with K-Nearest Neighbors (K-NN)

ML Theory Assignment – K-NN implemented in Python on a real-life example, run in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zfathcodes/exoplanet-knn-classification/blob/main/exoplanet_knn.ipynb)

## 1. Purpose

NASA's Kepler telescope flagged thousands of "Kepler Objects of Interest" (KOIs): stars that dim slightly, as if a planet were crossing in front of them. Many of these signals are **not** planets (eclipsing binary stars, instrument artifacts). Astronomers cannot follow up on every signal, so a model that separates real planets from false positives helps prioritize.

**Goal:** classify a KOI as a **confirmed planet (1)** or a **false positive (0)** from physical measurements, using K-NN, and show how the model was optimized and checked for overfitting/underfitting.

## 2. Dataset

* **Source:** NASA Exoplanet Archive, *Kepler Objects of Interest Cumulative Table* (doi:10.26133/NEA4).
* **Kaggle mirror:** "NASA Exoplanet Dataset" (arashnic).
* **Rows used:** only `CONFIRMED` and `FALSE POSITIVE`. `CANDIDATE` rows are unresolved, so they are held out and predicted at the end as a bonus.
* **Features (11):** orbital period, transit duration, transit depth, impact parameter, planet radius, equilibrium temperature, insolation flux, transit signal-to-noise, stellar temperature, stellar surface gravity, stellar radius.

### Data-leakage control
`koi_score`, `koi_pdisposition` and the `koi_fpflag_*` columns are outputs of the vetting process that decides the label. Using them would inflate accuracy without learning anything physical, so they are **excluded**. Error columns, IDs and names are excluded too.

## 3. How to run

1. Click the **Open in Colab** badge above (or upload `exoplanet_knn.ipynb` to Colab).
2. Choose **Runtime → Run all**.
3. The notebook downloads the table from the NASA Exoplanet Archive. If the download fails, upload the Kaggle CSV when prompted (or place it next to the notebook as `cumulative.csv`).

Local run: `pip install -r requirements.txt`, then open the notebook in Jupyter.

## 4. Steps in the notebook

| # | Step | Purpose |
|---|------|---------|
| 1 | Setup | Imports, fixed random seed (42) for reproducibility |
| 2 | Load data | NASA archive download with local/upload fallback |
| 3 | Labels & leakage control | Binary target, leaky columns removed |
| 4 | EDA | Class balance, missing values, feature distributions by class |
| 5 | Train/test split | Stratified 80/20; **test set sealed until the end** |
| 6 | Preprocessing pipeline | Median imputation, signed log transform for skewed features, standardization – all fitted on training folds only |
| 7 | Baselines | Majority class and logistic regression, so K-NN is judged against something |
| 8 | Optimization | Grid search over k, distance metric, weighting; feature-group ablation; PCA |
| 9 | Over/underfitting analysis | Train vs validation error and F1 against k |
| 10 | Final test evaluation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC and PR curves |
| 11 | **K-NN spotlight** | K-NN re-implemented from scratch in NumPy and checked against scikit-learn; decision boundaries for small, best and large k; neighbor-based explanation of a correct and a wrong prediction; scaling and noise-feature experiments; search-algorithm speed |
| 12 | Candidate predictions | Ranks unresolved candidates by neighbor-vote probability |
| 13 | Conclusions | Interpretation and limitations |

## 5. Optimization approach

K-NN learns no weights, so optimization means choosing the inputs and hyper-parameters that generalize best:

* **k:** odd values 1–51, selected by 5-fold stratified cross-validation on F1 of the planet class.
* **Distance metric:** Euclidean, Manhattan, Chebyshev.
* **Weighting:** uniform vs distance-weighted voting.
* **Scaling and log transform:** necessary because K-NN is distance-based and several features span orders of magnitude.
* **Feature ablation:** drop or keep each feature group (transit shape, planet, signal, star) to see which carry the signal.
* **PCA:** tested at several variance thresholds.

The notebook has no optimizer, activation function or data augmentation, because those belong to neural networks. The levers above are their K-NN equivalents.

## 6. K-NN spotlight

To show how K-NN itself works, not only how it scores:

* **From scratch:** a NumPy implementation (distance to every stored point, take the `k` closest, vote) that matches scikit-learn's predictions.
* **Decision boundary:** plotted on a 2-D PCA projection for `k = 1`, the best `k` and `k = 51`, showing jagged (overfit) versus smooth (underfit) boundaries.
* **Explainability:** each prediction is justified by listing and plotting its nearest neighbors, including a misclassified example.
* **Weaknesses:** performance with and without scaling, and as useless random features are added (compared with logistic regression).
* **Cost:** brute-force vs tree-based neighbor search time.

## 7. Overfitting / underfitting

Section 9 plots training vs validation **error rate (1 − accuracy, the zero-one loss)** and F1 against k:

* Small k (e.g. 1): near-zero training error but higher validation error → **overfitting**.
* Large k: both errors rise → **underfitting**.
* The selected k sits where validation performance is best.

## 8. Results

The following table shows the performance of the models on the held-out test dataset.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority Class | 0.6377 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.8353 | 0.7206 | 0.8909 | 0.7967 | 0.9169 |
| Baseline K-NN (k=5) | 0.9137 | 0.8606 | 0.9091 | 0.8842 | 0.9630 |
| **Optimized K-NN** | **0.9144** | **0.8535** | **0.9218** | **0.8864** | **0.9684** |

**Best model:** Optimized K-NN (k=11, Manhattan distance, uniform weights).

The optimized K-NN achieved an F1-score of 0.8864 and ROC-AUC of 0.9684. It performed slightly better than baseline K-NN, demonstrating the benefit of hyperparameter tuning.

## 9. Limitations

* Kepler-only data; labels come from a vetting process with some uncertainty.
* K-NN is sensitive to scaling and irrelevant features and slow on very large data.
* Predicted probabilities for candidates are neighbor-vote fractions, a ranking aid and not proof of a planet.

## 10. Demo

*(Optional)* Link to a short screen recording of the notebook running: `<add link>`

## 11. Citation

NASA Exoplanet Archive, *Kepler Objects of Interest Cumulative Table*, NExScI-Caltech/IPAC, doi:10.26133/NEA4.
