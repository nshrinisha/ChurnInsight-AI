# ChurnInsight AI: Predictive Analytics for Customer Retention in Banking 🚀

## Overview 🧠

ChurnInsight AI is a machine learning project designed to predict customer churn in the banking sector. The project aims to identify at-risk customers early, enabling banks to implement proactive retention strategies. By leveraging behavioral and demographic data, ChurnInsight achieves accurate predictions and provides actionable insights to reduce churn rates.

## Problem Statement 💡

Customer churn is a critical issue in the banking industry, directly impacting profitability. Retaining customers is more cost-effective than acquiring new ones, but traditional methods often fail to detect churn in time. This project addresses this gap using advanced machine learning techniques.

## Features ✨

- Predict customer churn using machine learning algorithms.
- Provide insights into key factors influencing churn. 🔍
- Handle class imbalance using synthetic oversampling (SMOTE).
- Compare model performance using metrics like F1 Score and ROC-AUC.

## Dataset 📊

The dataset, sourced from Kaggle, contains information on 10,000 customers from ABC Multinational Bank. It includes 14 attributes, such as:

- **Demographic Data:** Age, gender, etc.
- **Behavioral Data:** Balance, number of products, tenure, etc.
- **Outcome:** Whether the customer churned (binary classification).

### Data Preprocessing 🛠️

- **Data Cleaning:** Checked for null values and duplicates.
- **Outlier Handling:** Managed extreme values to improve model performance.
- **Class Balancing:** Applied SMOTE to address class imbalance.
- **Feature Engineering:** One-hot encoding for categorical features and normalization for numerical features.

## Machine Learning Pipeline 🏗️

### Models Implemented

1. **Logistic Regression:** Simple baseline for binary classification.
2. **k-Nearest Neighbors (k-NN):** Uses customer similarity for predictions.
3. **Decision Trees:** Captures non-linear patterns and provides interpretability.
4. **Random Forest:** Reduces overfitting using ensemble learning. 🌳
5. **XGBoost:** High-accuracy model with effective handling of imbalanced data. ⚡

### Evaluation Metrics 📈

- **F1 Score:** Measures the balance between precision and recall.
- **ROC-AUC:** Evaluates the model’s ability to distinguish between churned and non-churned customers.

## Results 🏆

- **Best Model:** Random Forest
  - **Validation F1 Score:** 0.72
  - **Validation ROC-AUC:** 0.80
- Other models like Logistic Regression and Decision Trees showed competitive performance but were outperformed by Random Forest.

## Challenges ⚠️

- **Class Imbalance:** Resolved using SMOTE.
- **Feature Selection:** Addressed using domain knowledge and exploratory data analysis.
- **Model Interpretability:** Enhanced through hyperparameter tuning and visualization.

## Future Work 🔮

- Enhance predictive accuracy through advanced feature engineering and optimization.
- Integrate real-time data streams for dynamic churn predictions. ⏱️
- Explore neural networks and sentiment analysis to capture complex patterns and customer behavior.

## How to Use 🛠️

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook:**
   Open the `Main.ipynb` file in Jupyter Notebook or any compatible IDE.

4. **Train the Models:**
   Follow the steps in the notebook to train and evaluate the models.

5. **Analyze Results:**
   Review the visualizations and metrics to understand model performance. 📊

## Directory Structure 📁

```
ChurnInsight-AI/
├── data/                # Dataset and processed data files
├── notebooks/           # Jupyter notebooks for EDA and modeling
├── results/             # Visualizations and performance metrics
├── README.md            # Project overview and instructions
├── requirements.txt     # Python dependencies
```

## Dependencies 🧩

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

## Authors ✍

- Aiswaryaa Velumani
- Sai Teja Aggunna
- Shrinisha Nirmalkumar
- Vaishnavi Pathipati

## Acknowledgments 

- **Dataset Source:** Kaggle

For further questions or contributions, feel free to open an issue or submit a pull request! ✨

