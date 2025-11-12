# project_Tripadvisor
High Rating Score Classification - Tripadvisor Restaurants

High Rating Score Classification
among Tripadvisor Restaurant Reviews
•	A classification model was developed to predict high rated restaurants in Tripadvisor reviews.
•	Feature importance was analyzed to identify the variables most strongly driving segment predictions.
•	XGBoost was selected as the primary model, as it provided a strong balance of interpretability and predictive accuracy. 
•	The models were implemented in Python using pandas, numPy and seaborn libraries among others, with encoded survey answers as the input variables.

Flat File Preparation and Feature Engineering
The dataset was transformed into a structured, flat file for machine learning, with numerical and categorical features. Key steps involved:
1.	Opening Hours and Time Features: Binary indicators for each day of the week and time-of-day features were created, preserving missing hours.
2.	Claim Status and Awards: Claimed status was converted into a binary indicator, and awards were expanded with binary columns for specific awards and derived features such as total awards and award score.
3.	Language Features: The primary language was encoded numerically, with hotspot features for English and all languages.
4.	Dietary and Price Features: Dietary options were turned into binary features, while price information was transformed into numeric features (price level and range).
5.	Restaurant Features and Meals: Meal types were converted into binary indicators, and a count of features per restaurant was added.
6.	Geolocation Features: Missing city names were filled using reverse geocoding.
7.	Final Outcome: The dataset was preprocessed with a combination of binary, categorical, and numeric features, ready for machine learning.
Exploratory Data Analysis (EDA) and Data Profiling
Key steps included:
•	Data Summary: A table was created to summarize each column's data type, missing values, and unique values.
•	Target Variable Encoding: A binary target variable based on the average rating was created to classify high (≥4.5) and low ratings.
•	Data Visualization: AutoViz was used for exploratory data visualization to better understand distributions.
Outlier Detection and Handling
Key methods:
1.	Shapiro-Wilk Test: A normality test identified non-normal columns for further processing.
2.	IQR Method: Outliers were detected using the Interquartile Range method, flagging values outside the range as outliers.
3.	Outlier Handling: Outliers were capped with NaN values, and their impact was assessed through correlation and distribution comparisons.
4.	Visualizing Missing Data: A missing data matrix was generated to analyze patterns.
Binary Columns Imputation using KNN
For binary columns, KNN imputation was used to fill missing values by considering the nearest neighbors' values:
•	KNN Process: The KNN imputer filled in missing data by predicting the majority value from the nearest neighbor, using Euclidean distance.
Continuous Columns Imputation using MICE
For continuous numeric columns, MICE (Multiple Imputation by Chained Equations) was used:
•	MICE Process: Missing values were imputed iteratively over 15 cycles using regression models.

Text Analysis 
Out of the open ended text objective quantitative indications were extracted as cuisine types, number of cuisines, special diets (vegetarian, vegan, Halal, Kosher and others). Subjective text was excluded to avoid data leakage.





Machine Learning Process for Predictive Modeling
Key steps:
1.	Data Preprocessing: Features were rescaled using Min-Max scaling to bring all numeric values within [0,1].
2.	Feature Selection: Lasso, Ridge, Gradient Boosting, and Random Forest models identified important features, with SelectKBest (Chi-squared test) and Recursive Feature Elimination (RFE) used for final feature selection.
3.	Model Training: Multiple models as Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost were trained using a 70% training, 15% validation, and 15% test split.
4.	Model Evaluation: XGBoost achieved the highest F1-Score (52.22%), Precision (63.80%), and Accuracy (70.87%).

Hyperparameter Tuning with RandomizedSearchCV
•	Goal: To improve the XGBoost model's performance.
•	Hyperparameter Tuning: RandomizedSearchCV tested 50 random combinations of hyperparameters (e.g., n_estimators, max_depth, learning_rate).
•	Best Hyperparameters: Subsample=0.8, reg_lambda=1, max_depth=3, n_estimators=200, learning_rate=0.05.
•	Model Performance: After tuning, XGBoost improved with a Precision of 0.72 (Class 0) and 0.65 (Class 1), a Recall of 0.88 (Class 0) and 0.38 (Class 1), and an F1-Score of 0.79 (Class 0) and 0.48 (Class 1).
Final Model Evaluation
•	Test Set Performance: Precision was 0.81 (Class 0), 0.48 (Class 1), and Recall was 0.54 (Class 0), 0.77 (Class 1).
•	Scale_pos_weight Impact: Adjusting scale_pos_weight improved recall for Class 1, indicating better sensitivity to the minority class.
•	Conclusion: The model has improved recall for Class 1 but could further benefit from refinement to reduce false positives.
________________________________________


Final Model Performance on Test Set:
•	Confusion Matrix: [[2058, 1741], [494, 1619]]
•	Accuracy: 62%
•	F1-Score (Class 0): 0.65, F1-Score (Class 1): 0.59

Awards, pecial diets, wide open hours and gourmet food found as related to high score in the restaurant review.
