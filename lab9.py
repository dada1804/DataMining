 #Reduced error pruning
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Assuming you have the dataset stored in a CSV file named 'dataset.csv'
dataset = pd.read_csv('Desktop/credit-g.csv')

label_encoder = LabelEncoder()
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = label_encoder.fit_transform(dataset[column])

# Splitting the dataset into input features (X) and target variable (y)
X = dataset.drop('class', axis=1)  # Replace 'credit_approval' with the actual target column name
y = dataset['class']  # Replace 'credit_approval' with the actual target column name

# Creating the Decision Tree model
model = DecisionTreeClassifier()

# Performing cross-validation and obtaining accuracy scores
accuracy_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation

# Reporting the accuracy scores for each fold
print("Accuracy Scores for each fold:", accuracy_scores)

# Calculating the average accuracy across all folds
average_accuracy = accuracy_scores.mean()

# Reporting the average accuracy
print("Average Accuracy:", average_accuracy)


# Creating the Decision Tree model with reduced error pruning
model = DecisionTreeClassifier(ccp_alpha=0.01)  # Adjust the value of ccp_alpha as per your dataset

# Performing cross-validation and obtaining accuracy scores with pruned model
accuracy_scores_pruned = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation with pruned model

# Reporting the accuracy scores for each fold with pruned model
print("Accuracy Scores for each fold (pruned):", accuracy_scores_pruned)

# Calculating the average accuracy across all folds with pruned model
average_accuracy_pruned = accuracy_scores_pruned.mean()

# Reporting the average accuracy with pruned model
print("Average Accuracy (pruned):", average_accuracy_pruned)

Output:
	Accuracy Scores for each fold: [0.675 0.705 0.65  0.695 0.73 ]
Average Accuracy: 0.691
