# DevelopersHub-Corporation-Data-Science-and-Analytics-Internship-Tasks
My data science and analytics internship tasks

# Task 1: Exploring and visualizing simple dataset

Objective:
Understand how to read, summarize, and visualize a dataset.

Dataset used: Iris

Tools: pandas, matplotlib and seaborn

Basic visualizations:
* Scatter plot to analyze relationships between variables.
* Histogram to examine data distributions.
* Box plot to detect outliers and spread of values.

Results:
There were 2 scatterplots in this task. From the first scatterplot, we see that:

* Setosa has smaller petal lengths and widths.
* Versicolor lies in the middle of the other two species in terms of petal length and width.
* Virginica has the largest petal lengths and widths.

The second scatterplot shows that:

* Setosa has smaller sepal lengths but larger sepal widths.
* Versicolor lies in the middle of the other two species in terms of sepal length and width.
* Virginica has larger sepal lengths but smaller sepal widths.

The histograms indicate that the sepal length and sepal width are normally distributed, whereas petal length and petal width diverge from normal distribution.

The graphs of boxplot shows that:

* Setosa has the smallest features and is less distributed with some outliers.
* Versicolor has the average features.
* Virginica has the highest features

# Task 2: Credit Risk prediction

Objective: 
Predict whether a loan applicant is likely to default on a loan. 

Dataset used: 
Loan Prediction Dataset

Instructions: 
*	Handling missing data appropriately. 
*	Visualizing key features such as loan amount, education, and income. 
*	Training a classification model like Logistic Regression or Decision Tree. 
*	Evaluating the model using accuracy and a confusion matrix.

Results:

1. Data distributions (Top part):

Loan Amount Distribution:

This histogram shows that the majority of loan amounts are concentrated in the lower range, with a long tail extending towards higher amounts. This suggests that smaller loans are more common in the dataset.

Education Levels:

This bar chart indicates that a significantly higher number of applicants are "Graduate" compared to "Not Graduate". This highlights an imbalance in the education distribution within the dataset.

Applicant Income Distribution:

This histogram, similar to "Loan Amount", shows that most applicants have lower incomes, with a few outliers having very high incomes.

2. Model performance - Logistic Regression:

Logistic Regression Accuracy (70.83%):

This is the overall accuracy score for the Logistic Regression model, meaning it correctly predicted the outcome (likely loan approval/disapproval) for 70.83% of the cases in the test set.

Confusion matrix of logistic regression:

True Label 0, Predicted Label 0 (Top-Left: 0):

True Negatives (TN): This indicates that the model correctly predicted 0 instances belonging to class 0 (e.g., loan disapproved) as class 0. This value being 0 means that the model never predicts class 0, or there are no instances of class 0 in the true labels within the test set.

True Label 0, Predicted Label 1 (Top-Right):

False Positives (FP): 28 instances that actually belonged to class 0 were incorrectly predicted as class 1. These are False Positives.

True Label 1, Predicted Label 0 (Bottom-Left: 0):

False Negatives (FN): This indicates that the model correctly predicted 0 instances belonging to class 1 as class 0. Similar to the top-left, this being 0 is problematic.

True Label 1, Predicted Label 1 (Bottom-Right):

True Positives (TP): 68 instances that actually belonged to class 1 were correctly predicted as class 1. These are True Positives.

3. Model performance - Decision tree:

Decision tree Accuracy (60.42%):

The overall accuracy of the Decision Tree model is 60.42%, which means that it correctly predicted the outcome in the train dataset.

Confusion matrix of decision tree:

True Label 0, Predicted Label 0 (Top-Left cell: 9):

True Negatives (TN): The model correctly predicted 9 instances as belonging to class 0.

True Label 0, Predicted Label 1 (Top-Right cell: 19):

False Positives (FP): The model incorrectly predicted 19 instances as belonging to class 1 when they were actually class 0.

True Label 1, Predicted Label 0 (Bottom-Left cell: 19):

False Negatives (FN): The model incorrectly predicted 19 instances as belonging to class 0 when they were actually class 1.

True Label 1, Predicted Label 1 (Bottom-Right cell: 49):

True Positives (TP): The model correctly predicted 49 instances as belonging to class 1.

Overall Summary and Comparison:

Data Characteristics: 

The dataset exhibits skewed distributions for loan amount and applicant income, and a significant imbalance in education levels, with "Graduate" being dominant. These characteristics can impact model performance, especially if not addressed.

Logistic Regression:

While it has a higher reported accuracy (70.83%), its confusion matrix reveals a critical flaw: it appears to be always predicting class 1. This suggests the model is not learning to distinguish between the two classes and is simply predicting the majority class, which can lead to misleadingly high accuracy if one class is heavily dominant. This model is essentially useless for making actual predictions if discriminating between classes is important.

Decision Tree:

The model achieved an accuracy of 60.42%, which means that 6 out of 10 predictions made by the Decision Tree model were correct. It correctly identified 49 positive cases (True Positives) and 9 negative cases (True Negatives).

However, it made a significant number of errors in its confusion matrix:

* It incorrectly classified 19 actual negative cases as positive (False Positives).
* It incorrectly classified 19 actual positive cases as negative (False Negatives).

Overall, an accuracy of 60.42% still indicates that the model has substantial room for improvement in its predictive capabilities. Further tuning, feature engineering, or exploring other models would likely be beneficial.

As for the classes 0 and 1, Label encoder is used in the source code mentioned above. It is necessary because in the cleaned train dataset, the column 'Education' has 2 outcomes: graduate or not graduate. The LabelEncoder converts these text categories into numerical labels (e.g., 'Graduate' → 0, 'Not Graduate' → 1). Machine learning models like Logistic Regression and Decision Trees generally require numerical input, so without this step, the training model would fail.

# Task 3: Customer Churn Prediction

Objective: 
 Identify customers who are likely to leave the bank. 

Dataset used: 
 Churn Modelling Dataset 

Instructions:

*	Cleaning and prepare the dataset.  
*	Encoding categorical features such as geography and gender. 
*	Training a classification model. 
*	Analyzing feature importance to understand what influences churn.

Results:
**Explanation of the output:**
1. We start by importing required libraries and load the dataset.

2. We clean and prepare the dataset. After dropping irrelevant columns, only meaningful numeric and categorical features remain. Our target variable is customers who are likely to leave a bank. The dataset of both features and target variable are displayed above. 

3. **Encoding output** 
We encode categorical features: Gender and Geography.

* Gender will be numeric (0 for female and 1 for male).
* Geography will show one-hot encoded columns like: Geography_Germany, Geography_Spain (France is dropped to avoid multicollinearity.)

4. **Class distribution(Target variable)**
The output of target variable is:
*  7963 customers stayed - 0 (Not churned)
*  2037 customers left - 1 (Churned)
5. **Confusion Matrix**
After training the model and predicting:
Interpretation:

* True Positives (185): Customers correctly predicted to leave.
* True Negatives (1538): Customers correctly predicted to stay.
* False Positives (55): Predicted leave, but stayed.
* False Negatives (222): Predicted stay, but actually left.

6. **Classification Report**
* Accuracy (~86%) means the model predicts customer behavior correctly about 86% of the time.
* Precision & recall for churned customers (label 1) are usually lower due to class imbalance.

7. **Feature Importance Plot**
Shows which features matter most:
* Age
* Balance
* Number of Products
* IsActiveMember
* Geography_Germany
* EstimatedSalary

Longer bars mean more influence on churn predictions.

**Summary:**
* Age and Balance often have the highest impact on whether a customer churns.
* Geography and IsActiveMember also play important roles.
* Model struggles slightly to detect customers who actually churn due to class imbalance.

# Task 4: Predicting Insurance Claim Amounts

Objective: 
 Estimate the medical insurance claim amount based on personal data.
 
Dataset used: 
 Medical Cost Personal Dataset 
 
Instructions: 
*	Training a Linear Regression model to predict charges. 
*	Visualizing how BMI, age, and smoking status impact insurance charges.  
*	Evaluating model performance using MAE and RMSE. 

# Task 5:

Objective: 
 Predict which customers are likely to accept a personal loan offer. 
 
Dataset used: 
 Bank Marketing Dataset (UCI Machine Learning Repository)
 
Instructions: 
*	Performing basic data exploration on features such as age, job, and marital status. 
*	Training a Logistic Regression or Decision Tree classifier. 
*	Analyzing the results to identify which customer groups are more likely to accept the offer. 
