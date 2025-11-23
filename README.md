# Machinelearningprojectmce
#This project aims to compare several Machine Learning models on two different datasets:
	1.	Spambase Dataset (UCI Machine Learning Repository)
	•	Binary classification: spam vs. not spam
	•	~5000 samples, 57 numerical features
	•	Relatively balanced
	2.	Diabetes Health Indicators Dataset (Kaggle)
	•	Binary classification: diabetes vs. no diabetes
	•	~250,000 samples, 21 numerical/ordinal features
	•	Highly imbalanced (≈ 85% class 0, 15% class 1)

The goal is to evaluate the performance of different models, understand their strengths and weaknesses, and discuss how dataset characteristics (size, imbalance, feature distribution…) influence model selection.
 Objectives :
	•	Clean and preprocess both datasets
	•	Handle imbalance where necessary
	•	Train multiple ML models
	•	Compare them using relevant metrics
	•	Analyze results and draw conclusions for each dataset

 Data Preprocessing

For both datasets:
	•	Removal of duplicates
	•	Removal of invalid rows (negative values)
	•	Standardization using StandardScaler
	•	Train-test split (80/20) with stratification

Additional for Diabetes Dataset:
	•	Dataset is extremely imbalanced
	•	Applied oversampling with SMOTE and threshold tuning
	•	Evaluated metrics beyond accuracy: recall, precision, F1-score, ROC-AUC

Models Implemented : 

We tested and compared the following models:

✔ Logistic Regression

✔ K-Nearest Neighbors (KNN)

✔ Random Forest

✔ Multi-Layer Perceptron (PyTorch)

✔ (Optional) Threshold optimization for KNN & Random Forest

Each model was evaluated on both datasets when appropriate.


Evaluation Metrics : 

Because the Diabetes dataset is imbalanced, we use more informative metrics:
	•	Accuracy
	•	Precision
	•	Recall (Sensitivity)
	•	F1-score
	•	Confusion Matrix
	•	ROC Curve & AUC Score
	•	Optimal probability threshold selection (when applicable)

For the Spam dataset (balanced), accuracy is more meaningful.

Main Results Summary:

1. Spambase Dataset
	•	Logistic Regression, KNN, and Random Forest all performed well
	•	Accuracy was high across models
	•	Dataset is balanced → classic ML methods work reliably

Best model:
➡ Logistic Regression or Random Forest (both ~94% accuracy)


2. Diabetes Dataset

A much more challenging dataset due to imbalance and noisy features.

Logistic Regression
	•	Recall class 1: 0.76 (best)
	•	Many false positives
	•	Good for detecting diabetics (high sensitivity)

KNN (optimal threshold)
	•	Accuracy: 0.76
	•	Better precision than Logistic Regression
	•	Balanced performance
	•	ROC-AUC: 0.74

Random Forest (optimal threshold)
	•	Accuracy: 0.81 (highest)
	•	But recall class 1: 0.30 (worst) → misses many diabetics
	•	Not suitable for minority detection

MLP (PyTorch)
	•	Best F1-score for class 1 (0.47)
	•	Good recall (0.64)
	•	Best balance overall

Best model:
➡ MLP (best balance)
➡ Logistic Regression (best recall)
