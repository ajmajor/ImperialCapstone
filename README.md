# Imperial AI & Machine Learning Capstone Project - Banking Customer Churn Detection  

Submitted by: Andrew Major

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT

The idea of the project is to produce a machine learning model that will help detect customers who are likely to churn. The strategy can be simply broken down into the following steps:

* **Data Collection**: Gather historical customer data, including features such as account balance, transaction frequency, customer demographics, etc.
* **Data Cleaning**: Handle missing values (impute or drop), remove irrelevant features, check for duplicates etc.
* **Data Exploration**: Understand the distribution of features and identify potential outliers ('rogue values').

* **Model Selection**: The following six alternative models were considered for use:

* Logistic Regression
* K-Nearest Neighbours
* Random Forest
* Gradient Boosting
* Support Vector Machines
* Neural Networks (Deep Learning)

The technical details of why these were selected and how they work need not be considered here; suffice to say that they are all good candidates for a binary classification task (i.e. predicting 'Yes' or 'No' to an outcome).

* **Performance Metrics**: We need to be able to compare the performance of these models in order to select the most suitable for the task. As we are aiming to predict a minority class (churners represent only 20% of our dataset), we are concerned with:

  * **Precision**: Proportion of actual churners out of all predicted churners.
  * **Recall (Sensitivity)**: Proportion of actual churners correctly predicted.
  * **F1-Score**: An aggregate measure combining precision and recall.

* **Data Preprocessing**:  Consider the nature of the data we have, and whether we need to combine or remove any before training our model.

* **Model Training and Evaluation**: Divide our data into a set for training the model, and a separate set to validate it afterwards (this checks how well the model works with data it has not been trained on).

* **Interpretability and Business Insights**:  Ultimately, we hope to be able to identify which categoreis of customer data are the strongest indicators of impending churn, and provide actionable insights to the business for remediation of churn.

## DATA

* The dataset is available for download on Kaggle [here](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset/data); it is made available under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

## MODEL  

A summary of the model you’re using and why you chose it.

## HYPERPARAMETER OPTIMSATION

The chosen hyperparameter optimisation strategy was to use the scikit-learn GridSearchCV and RandomizedGridSearchCV functions; although this is effectively a duplication of effort, I was interested to see how well the random search compared to a full grid search, both in terms of time and optimisation results. In the end, RandomizedSearchCV seemed to hit upon the same optimisation values as the brute force search, with reduced computing overheads (confirming what had been researched online).
The hyperparameters to be optimised, along with their eventual values are:

* **hyperparameter 1**: value one
* **hyperparameter 2**: value 2

## RESULTS

A summary of your results and what you can learn from your model

You can include images of plots using the code below:
![Screenshot](F1_Comparison.png)
