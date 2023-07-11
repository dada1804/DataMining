 import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Assuming you have the dataset stored in a CSV file named 'dataset.csv'
dataset = pd.read_csv('credit-g_csv.csv')

label_encoder = LabelEncoder()
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = label_encoder.fit_transform(dataset[column])


# Splitting the dataset into input features (X) and target variable (y)
X = dataset.drop('class', axis=1) 
y = dataset['class']  

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

Output:
Accuracy Scores for each fold: [0.65  0.735 0.665 0.71  0.725]
Average Accuracy: 0.697
