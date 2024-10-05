# 💳 Credit Card Fraud Detection

## Overview
This repository focuses on detecting fraudulent credit card transactions using **machine learning** techniques. With the rise of online transactions, it is critical to identify fraudulent activities to protect users from unauthorized charges. The **highly imbalanced dataset** (available [here](https://www.kaggle.com/mlg-ulb/creditcardfraud)) presents real-world challenges in the domain of fraud detection.

## 🧠 Dataset Details
The dataset consists of credit card transactions made by European cardholders in September 2013. It includes:
- **492 fraud cases** out of **284,807 transactions**, making up just **0.172%** of all data.
- The features are numerical, derived through **Principal Component Analysis (PCA)** for confidentiality.
  - **V1, V2, … V28**: Principal components from PCA.
  - **Time**: Time elapsed between transactions.
  - **Amount**: Transaction amount, useful for cost-sensitive analysis.
  - **Class**: Response variable (1 for fraud, 0 for non-fraud).

Given the class imbalance, traditional accuracy metrics are not useful. We recommend evaluating models using **Area Under the Precision-Recall Curve (AUPRC)**.

## 🛠 Key Features
1. **Data Preprocessing**: 
   - Handle class imbalance with techniques like **undersampling**, **oversampling**, and **SMOTE**.
   - Normalize features for efficient model training.

2. **Machine Learning Models**:
   - **Logistic Regression** 📉
   - **Random Forest** 🌲
   - **XGBoost** ⚡
   - **Support Vector Machines (SVM)** 🧪

3. **Evaluation Metrics**:
   - **Confusion Matrix**
   - **Precision-Recall AUC**
   - **F1-Score** and **ROC AUC**

## 🔍 How It Works
- **Step 1**: Load and preprocess the dataset using **Pandas**.
- **Step 2**: Explore data characteristics and visualize fraud vs non-fraud transactions.
- **Step 3**: Implement classification algorithms to detect fraud.
- **Step 4**: Evaluate model performance using **Precision-Recall AUC** and compare results.

## ⚙️ Installation and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Credit Card Fraud Detection.ipynb
   ```

## 📊 Results and Performance
- The model's performance is best measured using **Precision-Recall AUC** due to the dataset's heavy imbalance.
- Comparison of results from different machine learning models allows for insights into the most effective approach for fraud detection.

## 🧑‍💻 Prerequisites
- Python 3.x
- Libraries: `NumPy`, `Pandas`, `Scikit-learn`, `XGBoost`, `Matplotlib`

You can install dependencies with:
```bash
pip install -r requirements.txt
```

## 🎯 Key Takeaways
- Fraud detection is a **class imbalance** problem requiring specialized handling of minority class instances.
- The repository applies multiple machine learning techniques to detect fraud and evaluates them using metrics suited for imbalance, providing a **practical and scalable** solution for real-world fraud detection systems.

---

## 📄 License
This repository is licensed under the MIT License.

---

This project offers an extensive exploration of **machine learning techniques** applied to fraud detection, emphasizing practical approaches to imbalanced datasets. Feel free to contribute and enhance the detection methods for more robust results! 🚀
