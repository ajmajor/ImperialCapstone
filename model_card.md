# Model Card

See the [example Google model cards](https://modelcards.withgoogle.com/model-reports) for inspiration.

## Model Description

**Input:** The model takes tabular banking customer data in the form of:

* **RowNumber:** The sequential number assigned to each row in the dataset.
* **CustomerId:** A unique identifier for each customer.
* **Surname:** The surname of the customer.
* **CreditScore:** The credit score of the customer.
* **Geography:** The geographical location of the customer (e.g., country or region).
* **Gender:** The gender of the customer.
* **Age:** The age of the customer.
* **Tenure:** The number of years the customer has been with the bank.
* **Balance:** The account balance of the customer.
* **NumOfProducts:* The number of bank products the customer has.
* **HasCrCard:** Indicates whether the customer has a credit card (binary: yes/no).
* **IsActiveMember:** Indicates whether the customer is an active member (binary: yes/no).
* **EstimatedSalary:** The estimated salary of the customer.

**Output:** The model returns a binary classification of whether the customer will churn

**Model Architecture:** Describe the model architecture you’ve used

## Performance

The model is trained on the [Banking Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset/data) on Kaggle. The strategy was to make an initial stratified split of 80% to train the model with a 5-fold cross-validation, and validate the model using the remaining 20% hold-out split. The metrics used to assess model performance are:

* **Precision**: Proportion of true positive predictions among all positive predictions.
* **Recall (Sensitivity)**: Proportion of actual positives correctly predicted.
* **F1-Score**: Harmonic mean of precision and recall.

## Limitations

Outline the limitations of your model.

## Trade-offs

The model exhibits performance issues in general, due to the imbalanced nature of the training data.
