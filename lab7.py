import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
# Load the dataset from CSV
dataset = pd.read_csv('credit-g_csv.csv')
label_encoder = LabelEncoder()
for column in dataset.columns:
if dataset[column].dtype == 'object':
dataset[column] = label_encoder.fit_transform(dataset[column])
# Separate the features (X) and target variable (y)
X = dataset.drop('class', axis=1)
y = dataset['class']
# Create a decision tree classifier
decision_tree = DecisionTreeClassifier()
# Perform cross-validation and get the accuracy scores
cv_scores = cross_val_score(decision_tree, X, y, cv=5)
# Fit the decision tree on the entire dataset
decision_tree.fit(X, y)
# Print the decision tree
print(decision_tree)
# Print the cross-validation results
print("Cross-Validation Accuracy Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())


Output:
DecisionTreeClassifier()
Cross-Validation Accuracy Scores: [0.655 0.745 0.665 0.695 0.735]
Mean Accuracy: 0.699
