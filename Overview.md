# Imperial AI & Machine Learning Capstone Project  

Submitted by: Andrew Major

## 1. Problem Statement and Objective  

- **Problem**: Predict customer churn (i.e., which customers are likely to leave the bank).
- **Objective**: Develop a model that accurately predicts churn to help the bank retain valuable customers.

## 2. Data Preparation  

- **Data Collection**: Gather historical customer data, including features such as account balance, transaction frequency, customer demographics, etc.
- **Data Cleaning**:
-- Handle missing values (impute or drop).
-- Remove irrelevant features.
-- Check for duplicates.
- **Data Exploration**:
-- Understand the distribution of features.
-- Identify potential outliers.

## 3. Model Selection  

Consider the following four alternative models:

1. **Logistic Regression**:
-- A simple yet interpretable model.
-- Suitable for binary classification tasks.
-- May need input data converted to categorical (e.g., one-hot encoding for categorical features like gender, education level).

2. **Random Forest**:
-- Ensemble method combining multiple decision trees.
-- Handles non-linear relationships.
-- Can handle both numerical and categorical features.

3. **Gradient Boosting (e.g., XGBoost)**:
-- Powerful ensemble model.
-- Handles complex interactions.
-- Automatically handles missing values.
-- May not require explicit one-hot encoding.

4. **Neural Networks (Deep Learning)**:
-- Complex model capable of learning intricate patterns.
-- Requires substantial data and computational resources.
-- Can handle both numerical and categorical features.

## 4. Performance Metrics  

Choose appropriate metrics to compare model performance:

- **Accuracy**: Overall correctness of predictions.
- **Precision**: Proportion of true positive predictions among all positive predictions.
- **Recall (Sensitivity)**: Proportion of actual positives correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall.
- **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**: Measures overall model performance.

## 5. Data Preprocessing  

- **Feature Scaling**: Normalize numerical features (e.g., using Min-Max scaling or Z-score normalization).
- **Handling Categorical Features**:
-- Convert categorical features to numerical using one-hot encoding.
-- Consider label encoding for ordinal categorical features.
- **Feature Engineering**:
-- Create new features if relevant (e.g., customer tenure, transaction frequency).
-- Explore interactions between features.

## 6. Model Training and Evaluation  

- Split data into training and validation sets.
- Train each model using the chosen features.
- Evaluate models using the selected performance metrics.
- Tune hyperparameters (e.g., learning rate, tree depth) using cross-validation.

## 7. Interpretability and Business Insights  

- Interpret model coefficients (for logistic regression).
- Identify important features (e.g., feature importance in tree-based models).
- Provide actionable insights to the bank (e.g., which customer segments are most likely to churn).
