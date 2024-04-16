# Datasheet for Banking Customer Churn Capstone Project  

**Author:** Andrew Major

## Motivation  

* The dataset was created to be used for:
  * Exploratory data analysis to understand the factors influencing customer churn in banks.
  * Build machine learning models for predicting customer churn based on the given features.

* The dataset was created by Saurabh Badole, who is listed as the Owner.
* The organisation which funded the creation of the dataset is not identified.

## Composition  

* Each instances comprises the following columns:
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
  * **Exited:** Indicates whether the customer has exited the bank (binary: yes/no).
* There are 10,000 rows of data of which:
  * 7963 are non-churners, and
  * 2037 are churners (approximately 20%)
* The dataset is fully complete, and has no missing data in any columns.
* The data has not been fully anonymised,as it contains a Surname column, but the likelihood of being able to identify a person is slim without other pieces of information.

## Collection process

* The data collection methodology is listed as 'through Banking institutions'; no indication is given as to the provenance of the data, its original sources or the purpose behind the collection of the data. The geographical regions spanning the data are France, Germany and Spain.

## Preprocessing/cleaning/labelling

* The dataset has no missing values, but no indication of any pre-processing or cleaning is mentioned on the Kaggle website; there is likely to have been some cleaning performed, given the state of the data.
* There is no “raw” data saved in addition to the clean data, so it would not appear that support for unanticipated future uses was intended.

## Uses

* Given the small number of features, there are no other uses for which the dataset might be suitable which come to mind.
* No information is given about the way the dataset was collected and preprocessed/cleaned/labeled, and how that might impact future uses. There is potential that the dataset could result in unfair treatment of groups such as stereotyping in as much as there are columns indicating gender, bank balance and country of origin which might allow the unintended generalisation of data for those groups.
* There are no listed purposes on Kaggle for which the dataset should not be used; any use intended to generalise or discriminate against identifiable groupings of people based on country of residence, bank balance or gender should be strongly discouraged.

## Distribution

* The dataset is available for download on Kaggle [here](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset/data)
* The dataset is made available under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.  

## Maintenance

* The dataset is  provided 'as is', and the Expected Update Frequency is listed as 'Never'; however, it does show a history of updates since its initial upload.
