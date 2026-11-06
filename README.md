# Exoplanet Classification with K-Nearest Neighbors (K-NN)

**ML Theory Assignment 1 — Implementation and Optimization of K-NN using Python and Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zfathcodes/exoplanet-knn-classification/blob/main/exoplanet_knn.ipynb)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [How to Run the Project](#3-how-to-run-the-project)
4. [Implementation Workflow](#4-implementation-workflow)
5. [Optimization Methodology](#5-optimization-methodology)
6. [Optimization Results](#6-optimization-results)
7. [Overfitting and Underfitting Analysis](#7-overfitting-and-underfitting-analysis)
8. [Final Model Evaluation](#8-final-model-evaluation)
9. [Understanding K-NN: Practical Experiments](#9-understanding-k-nn-practical-experiments)
10. [Bonus: Unresolved Exoplanet Candidates](#10-bonus-unresolved-exoplanet-candidates)
11. [Conclusion](#11-conclusion)
12. [Limitations and Future Improvements](#12-limitations-and-future-improvements)
13. [Repository Structure](#13-repository-structure)
14. [References](#14-references)

---

## 1. Project Overview

NASA's Kepler space telescope identified thousands of objects known as *Kepler Objects of Interest (KOIs)*. These are stars whose brightness decreases periodically, potentially indicating that a planet is passing in front of them.

However, not every detected signal represents an actual planet. Some signals are caused by eclipsing binary stars, instrumental errors, or other astronomical phenomena.

Manually investigating every detected object is time-consuming. Machine learning can help distinguish likely planets from false-positive detections and prioritize objects for further investigation.

### Project Objective

The objective of this project is to build a **binary classification model using the K-Nearest Neighbors (K-NN) algorithm** to classify Kepler Objects of Interest into two categories:

- **Confirmed Planet (1):** An object classified as a confirmed exoplanet.
- **False Positive (0):** An object identified as a false planetary detection.

The classification is performed using 11 physical and observational features related to planetary characteristics, transit measurements, and host-star properties.

### Why K-NN?

K-NN is a supervised, distance-based machine learning algorithm. Instead of learning explicit model weights, it stores the training observations and classifies a new observation based on the majority class among its nearest neighbors.

This project explores not only the implementation of K-NN but also its optimization, performance, sensitivity to preprocessing, and ability to generalize to unseen observations.

---

## 2. Dataset

The project uses real astronomical observations from the **NASA Exoplanet Archive – Kepler Objects of Interest Cumulative Table**.

| Dataset Information | Details |
|---|---|
| Primary Source | NASA Exoplanet Archive |
| Dataset | Kepler Objects of Interest Cumulative Table |
| DOI | 10.26133/NEA4 |
| Kaggle Mirror | NASA Exoplanet Dataset by arashnic |
| Original Dataset Size | 9,564 rows × 153 columns |
| Confirmed Planets | 2,748 |
| False Positives | 4,839 |
| Unresolved Candidates | 1,977 |
| Classification Features | 11 |

Only confirmed planets and false positives are used for supervised training and evaluation.

The unresolved candidate observations are kept separate and used for an additional prediction experiment.

### Features Used

The model uses 11 numerical features divided into four groups.

| Feature Group | Features |
|---|---|
| Transit Shape | `koi_period`, `koi_duration`, `koi_depth`, `koi_impact` |
| Planet Characteristics | `koi_prad`, `koi_teq`, `koi_insol` |
| Signal Strength | `koi_model_snr` |
| Host Star Properties | `koi_steff`, `koi_slogg`, `koi_srad` |

These features represent orbital measurements, transit characteristics, planetary properties, signal strength, and stellar characteristics.

### Data Leakage Prevention

An important part of the project is preventing data leakage.

Columns such as `koi_score`, `koi_pdisposition`, and `koi_fpflag_*` are excluded because they contain information associated with NASA's vetting and classification process.

Including these columns could allow the model to rely on information closely related to the target label instead of learning meaningful patterns from physical measurements.

Additionally, identifiers, object names, and measurement error columns are excluded.

This ensures that the classifier primarily learns from the selected astronomical features.

---

## 3. How to Run the Project

The notebook is designed to run in Google Colab without requiring a local Python environment.

### Google Colab (Recommended)

1. Click the **Open in Colab** badge at the top of this README.
2. Open the notebook in Google Colab.
3. Select **Runtime → Run all**.
4. Allow the notebook to download the dataset and execute the analysis.

### Dataset Loading

The notebook attempts to download the cumulative table directly from the NASA Exoplanet Archive.

If the download is unsuccessful, it checks for a local file named `cumulative.csv`. If that is unavailable, it provides an option to upload the dataset manually.

### Local Execution

To run the notebook locally:

```bash
git clone https://github.com/zfathcodes/exoplanet-knn-classification.git

cd exoplanet-knn-classification

pip install -r requirements.txt

jupyter notebook exoplanet_knn.ipynb
```

A fixed random seed (`42`) is used for reproducibility. Since the NASA archive is periodically updated, results may vary slightly when the notebook is executed with a newer dataset version.

---

## 4. Implementation Workflow

The project follows a structured machine learning workflow, starting from data collection and ending with model evaluation and interpretation.

| Step | Implementation | Purpose |
|---|---|---|
| 1 | Environment Setup | Import libraries and initialize the random seed. |
| 2 | Dataset Loading | Retrieve the NASA dataset with fallback options. |
| 3 | Label Preparation | Create binary labels and separate unresolved candidates. |
| 4 | Exploratory Data Analysis | Examine class distribution, missing values, and feature distributions. |
| 5 | Train-Test Split | Perform a stratified 80:20 split. |
| 6 | Data Preprocessing | Apply median imputation, signed log transformation, and standardization. |
| 7 | Baseline Models | Evaluate majority-class and logistic regression baselines. |
| 8 | Model Optimization | Perform hyperparameter tuning, feature ablation, and PCA experiments. |
| 9 | Generalization Analysis | Examine training and validation performance across different values of k. |
| 10 | Final Evaluation | Calculate classification metrics and generate evaluation plots. |
| 11 | K-NN Experiments | Implement K-NN from scratch and investigate its behavior and limitations. |
| 12 | Candidate Prediction | Generate predictions for unresolved astronomical candidates. |
| 13 | Conclusion | Interpret the results and discuss limitations. |

### Data Preprocessing

K-NN relies on distance calculations. Therefore, features with larger numerical ranges can dominate the distance calculation if preprocessing is not performed.

The following preprocessing pipeline is used:

1. **Median Imputation:** Handles missing numerical values.
2. **Signed Log Transformation:** Reduces the effect of highly skewed feature distributions while preserving negative values where applicable.
3. **Standardization:** Places features on comparable scales.

All preprocessing operations are performed inside a Scikit-learn Pipeline, ensuring that preprocessing parameters are learned only from the relevant training data during cross-validation.

### Train-Test Split

The labelled dataset is divided using stratified sampling:

| Dataset | Number of Observations |
|---|---:|
| Training Set (80%) | 6,069 |
| Testing Set (20%) | 1,518 |
| Total | 7,587 |

The test set is kept separate from model selection and hyperparameter tuning to provide an independent final evaluation.

---

## 5. Optimization Methodology

Unlike neural networks, K-NN does not use trainable weights, gradient-based optimizers, or activation functions.

Its optimization focuses on selecting suitable input features, preprocessing methods, distance metrics, and hyperparameters to improve generalization.

The following techniques are investigated.

### 5.1 Hyperparameter Tuning

A Grid Search with 5-fold stratified cross-validation is performed to identify suitable K-NN configurations.

The following parameters are explored:

| Parameter | Values Tested |
|---|---|
| Number of Neighbors (k) | Odd values from 1 to 51 |
| Distance Metric | Euclidean, Manhattan, Chebyshev |
| Voting Strategy | Uniform and Distance-Weighted |

The configurations are evaluated using the F1-score of the confirmed-planet class.

### 5.2 Feature Group Ablation

Feature ablation is performed to understand how different astronomical measurements contribute to classification performance.

Each feature group is removed individually and evaluated using cross-validation.

Experiments are also conducted using individual feature groups to understand how much predictive information each group contains independently.

### 5.3 Dimensionality Reduction with PCA

Principal Component Analysis (PCA) is tested to investigate whether reducing the feature dimensions improves K-NN performance.

Three variance-retention levels are evaluated:

- 99%
- 95%
- 90%

The results are compared against the model using all original features without PCA.

### 5.4 Preprocessing Optimization

Three preprocessing configurations are compared:

- No scaling.
- Standardization only.
- Signed log transformation followed by standardization.

This experiment demonstrates the importance of feature scaling for distance-based algorithms.

### 5.5 Overfitting and Underfitting Analysis

Training and validation performance are examined across different values of k.

Both error rate and F1-score are plotted to identify the relationship between model complexity and generalization.

### 5.6 Why Traditional Neural Network Optimization Techniques Are Not Used

Optimizers and activation functions are not applicable to K-NN because it does not learn parameters through iterative gradient-based training.

Instead, hyperparameter tuning, feature selection, preprocessing optimization, and cross-validation are used as appropriate optimization techniques for this algorithm.

---

## 6. Optimization Results

The following results represent the mean scores obtained through 5-fold cross-validation on the training dataset.

### 6.1 Baseline Comparison

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---:|---:|---:|
| Majority Class | 0.6378 | 0.0000 | 0.5000 |
| Logistic Regression | 0.8255 | 0.7905 | 0.9095 |
| Baseline K-NN (k=5) | 0.9011 | 0.8691 | 0.9542 |

K-NN performs better than both baseline approaches, demonstrating its ability to identify nonlinear patterns in the selected feature space.

### 6.2 Distance Metric and Voting Strategy Comparison

The best cross-validation F1-score obtained for each combination is:

| Distance Metric | Distance-Weighted | Uniform |
|---|---:|---:|
| Chebyshev | 0.8642 | 0.8624 |
| Euclidean | 0.8755 | 0.8751 |
| Manhattan | 0.8758 | **0.8759** |

**Best Configuration:**

- Number of neighbors: `k = 11`
- Distance metric: Manhattan
- Voting strategy: Uniform
- Cross-validation F1-score: `0.8759`

Manhattan and Euclidean distances produce very similar results, suggesting that the choice between these metrics has a relatively small impact on this dataset.

### 6.3 Feature Group Ablation Results

| Feature Configuration | CV F1-Score | CV Accuracy |
|---|---:|---:|
| All Features | **0.8759** | **0.9051** |
| Without Transit Shape | 0.8391 | 0.8791 |
| Without Planet | 0.8509 | 0.8868 |
| Without Signal | 0.8573 | 0.8901 |
| Without Star | 0.8609 | 0.8939 |
| Only Transit Shape | 0.8117 | 0.8562 |
| Only Planet | 0.7518 | 0.8155 |
| Only Signal | 0.5722 | 0.6883 |
| Only Star | 0.5959 | 0.7184 |

**Observation:** Transit-shape measurements provide the strongest individual feature group. However, removing any feature group reduces performance, indicating that all four groups contribute useful information.

### 6.4 PCA Comparison

| Configuration | CV F1-Score | CV Accuracy |
|---|---:|---:|
| No PCA | **0.8759** | **0.9051** |
| PCA – 99% Variance | 0.8697 | 0.9005 |
| PCA – 95% Variance | 0.8389 | 0.8797 |
| PCA – 90% Variance | 0.8350 | 0.8769 |

PCA does not improve the classification performance. Retaining all original features produces the best results in these experiments.

### 6.5 Preprocessing Comparison

| Preprocessing Method | CV Accuracy | CV F1-Score | ROC-AUC |
|---|---:|---:|---:|
| No Scaling | 0.8159 | 0.7593 | 0.8889 |
| Standardization Only | 0.8654 | 0.8275 | 0.9325 |
| Log Transform + Standardization | **0.9051** | **0.8759** | **0.9610** |

The combination of log transformation and standardization produces the best performance.

This highlights how strongly K-NN depends on appropriate feature preprocessing.

---

## 7. Overfitting and Underfitting Analysis

Overfitting and underfitting are investigated by examining training and validation performance for different values of k.

Since K-NN does not optimize a training loss over epochs, the notebook uses the classification error rate (zero-one loss) and F1-score as evaluation measures.

### Observations

**1. Small k – Overfitting**

When `k = 1`, the model relies on a single nearest neighbor.

- Training error is nearly zero.
- Validation error is comparatively higher.
- Train-validation accuracy gap is approximately 0.119.

This indicates overfitting because the model becomes highly sensitive to individual training observations.

**2. Selected k – Better Generalization**

At `k = 11`:

- The train-validation accuracy gap decreases to approximately 0.013.
- Validation performance is near its optimum.
- Predictions are less sensitive to individual observations.

**3. Large k – Underfitting**

When k becomes very large, the model considers too many neighbors.

- Local patterns become less influential.
- Predictions become increasingly generalized.
- Both training and validation errors increase.

The selected value of `k = 11` provides a better balance between model complexity and generalization.

---

## 8. Final Model Evaluation

After hyperparameter selection, the models are refitted using the complete training dataset and evaluated on the held-out test dataset containing 1,518 observations.

### 8.1 Test Set Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority Class | 0.6377 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression | 0.8353 | 0.7206 | 0.8909 | 0.7967 | 0.9169 |
| Baseline K-NN (k=5) | 0.9137 | 0.8606 | 0.9091 | 0.8842 | 0.9630 |
| **Optimized K-NN (k=11)** | **0.9144** | **0.8535** | **0.9218** | **0.8864** | **0.9684** |

The optimized K-NN configuration uses Manhattan distance and uniform voting.

It achieves the highest F1-score and ROC-AUC among the evaluated models.

### 8.2 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| False Positive | 0.95 | 0.91 | 0.93 | 968 |
| Confirmed Planet | 0.85 | 0.92 | 0.89 | 550 |

The model successfully identifies approximately 92% of the actual confirmed planets in the test set.

### 8.3 Confusion Matrix

| Actual / Predicted | False Positive | Confirmed Planet |
|---|---:|---:|
| False Positive | 881 | 87 |
| Confirmed Planet | 43 | 507 |

**Interpretation:**

- 881 false positives are correctly classified.
- 507 confirmed planets are correctly classified.
- 87 false positives are incorrectly classified as planets.
- 43 confirmed planets are incorrectly classified as false positives.

The confusion matrix demonstrates that the model performs well on both classes while maintaining a high recall for confirmed planets.

### 8.4 Performance Interpretation

The optimized K-NN model achieves an F1-score of 0.8864, compared with 0.7967 for logistic regression.

However, the improvement over the baseline K-NN model is relatively small:

- Baseline K-NN F1: 0.8842
- Optimized K-NN F1: 0.8864

This indicates that preprocessing and feature selection contribute more substantially to performance than fine-tuning the hyperparameters alone.

The difference between the two K-NN configurations should also be interpreted cautiously because small score improvements may fall within sampling variation.

The notebook additionally includes confusion matrix visualization, ROC curves, and precision-recall curves for further evaluation.

---

## 9. Understanding K-NN: Practical Experiments

Beyond model optimization, several experiments are performed to understand how K-NN works internally and where its limitations arise.

### 9.1 K-NN Implementation from Scratch

K-NN is implemented using NumPy without relying on Scikit-learn's KNeighborsClassifier.

The implementation follows three main steps:

1. Calculate the distance between the new observation and every training observation.
2. Identify the k nearest neighbors.
3. Predict the class using majority voting.

The from-scratch implementation is compared with Scikit-learn using the same preprocessed data, Euclidean distance, and uniform voting with `k = 11`.

Both implementations achieve 0.9216 accuracy and agree on 100% of the test predictions.

This experiment demonstrates that the underlying K-NN logic has been implemented correctly.

*Note: This is a separate Euclidean-distance implementation experiment, not the final optimized Manhattan-distance configuration.*

### 9.2 Decision Boundary Visualization

To understand how the number of neighbors affects classification boundaries, the original 11-dimensional feature space is projected onto two principal components using PCA.

Decision boundaries are visualized for:

- `k = 1`: Highly irregular boundaries that demonstrate overfitting.
- `k = 11`: A more balanced classification boundary.
- `k = 51`: Smoother boundaries that demonstrate the effect of excessive generalization.

These visualizations help explain how the choice of k affects model complexity.

### 9.3 Neighbor-Based Prediction Explanation

One advantage of K-NN is that its predictions can be explained by examining the nearest training observations.

The notebook investigates one correctly classified observation and one misclassified observation.

For the selected misclassified false positive, 9 out of its 11 nearest neighbors belong to the confirmed-planet class.

Consequently, the model predicts it as a planet.

This demonstrates how K-NN makes decisions based on local similarity in feature space.

### 9.4 Effect of Irrelevant Features

K-NN relies on distance calculations, which makes it sensitive to irrelevant input features.

To investigate this limitation, random noise features are appended to the dataset and the performance of K-NN is compared with logistic regression.

| Number of Noise Features | K-NN F1 | Logistic Regression F1 |
|---:|---:|---:|
| 0 | 0.8759 | 0.7905 |
| 5 | 0.8271 | 0.7892 |
| 10 | 0.7910 | 0.7865 |
| 20 | 0.7637 | 0.7883 |
| 40 | 0.7246 | 0.7853 |

As irrelevant features increase, K-NN performance decreases considerably.

Logistic regression is comparatively less affected in this experiment.

This highlights the importance of feature selection when using distance-based algorithms.

### 9.5 Computational Cost

K-NN stores training observations and performs most of its computational work during prediction.

The notebook compares three neighbor-search algorithms.

| Search Algorithm | Fit Time (s) | Prediction Time (s) |
|---|---:|---:|
| Brute Force | 0.0198 | 0.2212 |
| KD-Tree | 0.0344 | 0.2279 |
| Ball Tree | 0.0359 | 0.2326 |

For this relatively small, 11-dimensional dataset, tree-based methods do not provide a meaningful speed improvement over brute-force search.

This illustrates one of K-NN's limitations: prediction can become computationally expensive as the dataset grows.

---

## 10. Bonus: Unresolved Exoplanet Candidates

The original dataset also contains 1,977 observations labelled as `CANDIDATE`.

These objects are not used for supervised training or test evaluation because their final classifications are unresolved.

After training, the optimized K-NN model is used to examine these observations.

For each candidate, a neighbor-vote score is calculated based on the proportion of its 11 nearest neighbors that belong to the confirmed-planet class.

### Results

- Total unresolved candidates: 1,977
- Candidates with a neighbor-vote score of at least 0.5: 859

The resulting scores can be used to rank candidates for further investigation.

However, these scores are not calibrated probabilities and should not be interpreted as scientific confirmation of an exoplanet.

Multiple candidates may also receive identical scores, including a score of 1.0, making their relative ranking arbitrary.

---

## 11. Conclusion

This project demonstrates the implementation, optimization, and evaluation of the K-Nearest Neighbors algorithm using a real-world astronomical dataset.

The key findings are:

1. **K-NN outperforms the baseline models:** The optimized configuration achieves a test F1-score of 0.8864 and ROC-AUC of 0.9684, outperforming logistic regression.

2. **Preprocessing significantly improves performance:** Combining signed log transformation with standardization improves cross-validation F1 from 0.7593 to 0.8759.

3. **Transit-shape measurements are particularly informative:** Feature ablation indicates that transit-related measurements contribute strongly to classification performance.

4. **PCA does not improve the model:** Retaining the original feature set produces better results than the tested dimensionality-reduction configurations.

5. **Hyperparameter tuning provides a relatively small improvement:** The optimized model performs slightly better than the baseline K-NN model, suggesting that preprocessing and feature selection have a greater impact.

6. **The selected value of k balances overfitting and underfitting:** Small values of k produce unstable decision boundaries, while excessively large values lead to underfitting.

7. **K-NN is interpretable but sensitive to its feature space:** Predictions can be explained through nearest neighbors, but irrelevant features and inconsistent scaling can significantly affect performance.

Overall, the project demonstrates how a relatively simple distance-based algorithm can perform effectively on a real-world classification problem when supported by appropriate preprocessing, feature analysis, and systematic evaluation.

---

## 12. Limitations and Future Improvements

### Limitations

Despite the encouraging results, the project has several limitations:

- **Dataset scope:** The analysis is limited to observations from the Kepler mission and does not establish how well the model generalizes to other telescope datasets.
- **Label uncertainty:** The original classifications are based on an astronomical vetting process and may contain uncertainty.
- **Repeated host stars:** A random train-test split may place different KOIs belonging to the same host star in both training and testing sets. A star-grouped split was not performed, so the reported scores may be optimistic.
- **Sensitivity to features:** K-NN is sensitive to feature scaling and irrelevant dimensions.
- **Computational complexity:** Prediction becomes more expensive as the number of training observations increases.
- **Candidate predictions:** Neighbor-vote fractions are ranking scores rather than calibrated probabilities.
- **Dataset updates:** The NASA archive is periodically updated, which may cause slight differences in results across executions.

### Future Improvements

Potential extensions include:

- Using star-grouped cross-validation to reduce possible data leakage between related observations.
- Investigating class-balancing techniques.
- Exploring metric-learning methods such as Neighborhood Components Analysis.
- Comparing K-NN with additional tree-based classification algorithms.
- Testing model generalization on observations from the TESS mission.
- Investigating more robust approaches to feature selection and dimensionality reduction.

---

## 13. Repository Structure

```text
exoplanet-knn-classification/
│
├── README.md
│   └── Project documentation
│
├── exoplanet_knn.ipynb
│   └── Complete implementation and analysis
│
└── requirements.txt
    └── Required Python dependencies
```

---

## 14. References

1. NASA Exoplanet Archive, *Kepler Objects of Interest Cumulative Table*, NExScI-Caltech/IPAC.

   DOI: https://doi.org/10.26133/NEA4

2. Kaggle, *NASA Exoplanet Dataset*, by arashnic.

3. Scikit-learn Documentation, *Nearest Neighbors Classification*.

   https://scikit-learn.org/stable/modules/neighbors.html

4. Scikit-learn Documentation, *Model Selection and Grid Search*.

   https://scikit-learn.org/stable/modules/grid_search.html

---

**Project:** Exoplanet Classification with K-Nearest Neighbors (K-NN)  
**Implementation:** Python | NumPy | Pandas | Scikit-learn | Matplotlib | Seaborn  
**Environment:** Google Colab / Jupyter Notebook  
**Dataset Source:** NASA Exoplanet Archive
