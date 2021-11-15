# Project proposal for *YOUR PROJECT TITEL*
Author: *Graydon Hall*

## 1. Why: Question/Topic being investigated 1pt
I'm going to take famous titanic dataset from this kaggle competition. I will explore the following 4 questions:
1. Which model will produce the best results on the testing data
2. Investigating the use of PCA to try and visualize
3. Which featuers are most predictive in this model, and which can be ignored?
4. What is the best strategy for handling the 177 null values which show up in the age column

## 2. How: Plan of attack 1pt
0. Function definitions - define useful functions to help me throughout the lab
1. Load the titanic dataset 
    - from the kaggle completition
2. Inspect the data through visualizations 
    - Explore which features are useful and which can be dropped
    - Try to find visual trends in the features
3. Preprocessing
    - encode gender to make it numerical
    - either drop null ages or replace them with mean of the column
4. Create training and test sets
5. Compare models using cross-validation
    - Compare `LogisticRegression()`, `SVC()`, `BernoulliNB()`, `RandomForestClassifier()`, `GradientBoostingClassifier()` using cross validation
6. Hyperparameter tuning using grid search
    - Perform tuning on 3 models
7. Testing out CatBoost, XGBoost, and our best tuned model (Gradient boosting classifier)
    - Take top 2 tuned models, and CatBoost and XGBoost and put them all head to head
8. Analysis on how to handle null ages
    - Based on running steps 5 and 7 with 2 strategies (dropping null age values and replacing them with the mean age) which strategy yields best results
9. Visualzing the data using PCA
    - Attept to find useful visulaization using PCA
10. Feature Importances
    - Determine the importance of each feature

## 3. What: Dataset, models, framework, components 2pts
    -Dataset: Titanic Dataset
    -Models: `LogisticRegression()`, `SVC()`, `BernoulliNB()`, `RandomForestClassifier()`, `GradientBoostingClassifier()`, `CatBoost`, `XGBoost`
    -Frameworks: sklearn, xgboost, catboost, pandas, seaborn, matplotlib, 
    -Components: ?