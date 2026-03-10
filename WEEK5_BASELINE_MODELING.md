# Week 5 Baseline Modeling

## Scope
Baseline modeling was executed on the currently available QSAR biodegradation dataset.

## Constraint
The requested degradation-rate regression task is not possible with this dataset because the target is a binary biodegradation class rather than a continuous rate constant.

## Training Setup
- Train/test split: 80/20, stratified, random_state=42
- Baselines run: Logistic Regression, Random Forest Classifier
- Features: 41 numeric descriptor columns
- Target: binary biodegradation class

## Results
### logistic_regression
- accuracy: 0.8626
- macro_f1: 0.8456
- roc_auc: 0.9136

### random_forest_classifier
- accuracy: 0.8768
- macro_f1: 0.8529
- roc_auc: 0.9423

## Interpretation
These baselines validate the project pipeline and show the descriptor set is predictive for class-based biodegradation outcomes.

## To Reach the Original Week 5 Goal
Acquire a dataset with continuous degradation-rate targets such as half-life or rate constant values. Once available, the same project structure can be extended to run Linear Regression and Random Forest Regressor baselines with MAE, RMSE, and R2.
