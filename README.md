# Phase3project
H1N1 and Seasonal Flu Vaccine Prediction

Overview
This project aims to predict whether individuals received the H1N1 flu vaccine using a dataset from the 2009 National H1N1 Flu Survey. The project utilizes two machine learning models: Logistic Regression and Decision Tree Classifier to make predictions. The goal is to help public health efforts by identifying factors that influence vaccination decisions and guiding future vaccination campaigns.
Business Understanding
Background
In light of the COVID-19 pandemic, understanding factors that drive vaccination behavior has become more critical. By studying individuals’ backgrounds, opinions, and health behaviors, we can predict their likelihood of getting vaccinated. This analysis focuses on predicting whether an individual received the H1N1 vaccine, using data collected from the 2009 National H1N1 Flu Survey.
Problem Statement
The challenge is to predict whether people received the H1N1 flu vaccine using the data from the survey. This binary classification problem requires identifying relevant features, building a predictive model, and evaluating its performance to support public health efforts in future vaccination campaigns.
Objectives
1.	Data Cleaning & Preprocessing:
o	Handle missing values.
o	Convert categorical features to numerical values.
2.	Model Development:
o	Build two machine learning models: Logistic Regression and Decision Tree Classifier.
3.	Model Evaluation:
o	Assess the models’ performance using accuracy, precision, recall, and F1 score.
4.	Insights:
o	Identify important features influencing the vaccination decision.
Data Understanding
The dataset used in this project is from the 2009 National H1N1 Flu Survey and contains the following types of features:
•	Demographic Features: Age, Education, Race, Income, Marital Status, etc.
•	Health Behavior Features: Data on behaviors such as wearing face masks, avoiding large gatherings, taking antiviral meds, etc.
•	Health Information: Opinions on vaccine effectiveness, perceived risks of H1N1, chronic health conditions, healthcare access, and more.
The target variable is whether the respondent received the H1N1 flu vaccine, which is a binary outcome (1 = Vaccinated, 0 = Not Vaccinated).
Data Preparation
Steps:
1.	Handling Missing Data:
o	We imputed missing values for numerical features using the mean strategy.
o	Categorical variables were imputed with an "Unknown" placeholder.
2.	Feature Engineering:
o	Categorical variables were encoded using OneHotEncoder to convert them into a numerical format that can be used by the models.
3.	Feature Selection:
o	We selected a subset of relevant features based on domain knowledge, correlation analysis, and data exploration.
4.	Balancing the Dataset:
o	The dataset was imbalanced, with a much larger proportion of individuals who did not receive the vaccine. We used oversampling (SMOTE) to balance the dataset and address this issue.
Modeling
Logistic Regression Model
We used Logistic Regression as a baseline model due to its simplicity and interpretability. Logistic regression is a probabilistic model that estimates the likelihood of the target variable (H1N1 vaccination) based on the predictor features. The model is trained on the preprocessed dataset, and we evaluate it using accuracy, precision, recall, and F1 score.
Decision Tree Classifier Model
A Decision Tree Classifier was used to capture more complex relationships in the data. Decision Trees are non-linear models that split the data into subsets based on feature values. This model allows for easy interpretation and visualization, helping to understand which features are most important for the prediction.
Evaluation
Model Performance:
We used the following metrics to evaluate both models:
1.	Accuracy: The percentage of correct predictions.
2.	Precision: The proportion of positive predictions that are actually correct.
3.	Recall: The proportion of actual positive instances that are correctly predicted.
4.	F1 Score: The harmonic mean of precision and recall, providing a balance between the two.
Insights from Evaluation:
•	The models show decent accuracy but struggle with identifying vaccinated individuals (low precision and recall for the vaccinated class).
•	The imbalance in the dataset affects the model’s ability to predict the minority class (vaccinated individuals).
•	Both models have similar performance, but the decision tree provides more interpretability, allowing us to visualize the decision-making process.
Results Interpretation
The Logistic Regression and Decision Tree Classifier models performed similarly, with an accuracy of around 73%. However, the models struggled to predict the vaccinated class, as shown by their low precision and recall for this class. This suggests that additional work is needed, such as improving class balance or using more advanced models like Random Forest or Gradient Boosting.
Next Steps
1.	Hyperparameter Tuning:
o	Adjust parameters such as max_depth and min_samples_split in the Decision Tree to improve performance.
2.	Address Class Imbalance:
o	Further balance the dataset using SMOTE or undersampling techniques.
3.	Try Advanced Models:
o	Implement Random Forest, Gradient Boosting, or XGBoost to see if they outperform the current models.
4.	Feature Importance:
o	Use feature importance analysis to identify and select the most relevant features for future models.
Usage
1.	Clone this repository to your local machine.
2.	Install the required dependencies
Conclusion
This project aimed to predict H1N1 vaccination using machine learning models. Despite achieving a decent accuracy of 73%, the model struggled with predicting vaccinated individuals due to class imbalance. Further improvements in data preprocessing, model selection, and hyperparameter tuning are necessary to achieve better predictions.
Acknowledgments
•	Dataset: National 2009 H1N1 Flu Survey
•	Libraries Used: pandas, scikit-learn, matplotlib, numpy


