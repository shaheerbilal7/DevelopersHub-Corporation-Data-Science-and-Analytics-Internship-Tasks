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

Results:
Here's a breakdown of what each results indicate:

1. Data Visualization:

These plots show the relationship between different features (age, job, marital status) and the "deposit" variable (presumably whether a deposit was made or a loan was accepted).

* Age Distribution by Deposit observation(Top plot): 

There appears to be a higher concentration of "no" deposits/loans among younger individuals (around 30-40 years old), while the "yes" group seems to be more spread out or slightly older on average. It also suggests that the majority of the customers are within the 20 to 60 age range.

* Job vs Deposit observation (middle plot): 

Certain job types, like "management" and "technician," have a significant number of both "yes" and "no" deposits. "Retired" individuals seem to have a relatively higher proportion of "yes" deposits compared to "no." "Students" and "unemployed" individuals have fewer total instances, but it's worth noting their deposit behavior.

* Marital Status vs Deposit observation (bottom plot): 

"Married" individuals constitute the largest group, with more "no" deposits than "yes." "Single" individuals also show a considerable number of deposits, but with a more balanced ratio compared to "Married". "Divorced" individuals have the fewest instances.

2. Machine Learning Model Results:

This section presents the performance metrics for two machine learning models: Logistic Regression and Decision Tree.

Logistic Regression Results:

* Confusion Matrix: [[911 255]] and [[259 888]] : 
These likely represent the true negatives, false positives, false negatives, and true positives.

    * For class 0 (no deposit/loan): 911 correctly predicted as "no", 255 incorrectly predicted as "yes" (false positives).
    * For class 1 (deposit/loan): 259 incorrectly predicted as "no" (false negatives), 888 correctly predicted as "yes".

* Precision, Recall, F1-score for class 0 (no): 0.78, 0.80, 0.79 respectively. Support: 1166.
* Precision, Recall, F1-score for class 1 (yes): 0.77, 0.77, 0.77 respectively. Support: 1067.
* Accuracy: 0.78
* Macro Avg: Precision 0.78, Recall 0.78, F1-score 0.78.
* Weighted Avg: Precision 0.78, Recall 0.78, F1-score 0.78.

Decision Tree Results:

* Confusion Matrix: [[867 299]] and [[136 931]] :

    * For class 0 (no deposit/loan): 867 correctly predicted as "no", 299 incorrectly predicted as "yes" (false positives).
    * For class 1 (deposit/loan): 136 incorrectly predicted as "no" (false negatives), 931 correctly predicted as "yes".

* Precision, Recall, F1-score for class 0 (no): 0.86, 0.74, 0.80 respectively. Support: 1166.
* Precision, Recall, F1-score for class 1 (yes): 0.76, 0.87, 0.81 respectively. Support: 1067.
* Accuracy: 0.81
* Macro Avg: Precision 0.81, Recall 0.81, F1-score 0.81.
* Weighted Avg: Precision 0.81, Recall 0.80, F1-score 0.81.

Summary and Interpretation:

1. Exploratory Data Analysis (EDA): The visualizations provide initial insights into the relationships between demographic/job features and the target variable (deposit/loan acceptance). This is crucial for understanding the data and guiding feature engineering.

2. Model Performance:

* The Decision Tree model appears to outperform the Logistic Regression model in terms of overall accuracy (0.81 vs 0.78). It also shows slightly better F1-scores for both classes, indicating a better balance between precision and recall.
* The Logistic Regression convergence warning is a critical point. Its results might not be optimal because the optimization algorithm didn't fully converge. It could potentially perform better with more iterations or if the data were scaled (e.g., using StandardScaler or MinMaxScaler).

3. Class Imbalance (Potential): 

While not explicitly stated as an issue, the "support" values for class 0 (1166) and class 1 (1067) are relatively close. However, for some categorical features in the visualizations, there might be imbalances in the "yes" vs. "no" counts, which could affect model performance on less represented classes.
