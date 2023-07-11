import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
data = pd.read_csv('credit-g.csv')
for column in data.columns:
if data[column].dtype == 'object':
data[column] = labelencoder.fit_transform(data[column])
# Load the dataset
# Replace 'your_dataset.csv' with the actual file name and path
# Split the dataset into features (X) and labels (y)
X = data.drop('class', axis=1) # Replace 'label' with the column name of your labels
y = data['class']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Create an SVM classifier
svm = SVC()
# Train the SVM classifier
svm.fit(X_train, y_train)
# Make predictions on the test set
y_pred = svm.predict(X_test)
# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
decision_tree = DecisionTreeClassifier()
# Train the Decision Tree classifier
decision_tree.fit(X_train, y_train)
# Make predictions on the test set using Decision Tree
y_pred_dt = decision_tree.predict(X_test)
# Calculate the accuracy of Decision Tree
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print('Decision Tree Accuracy:', accuracy_dt)

Output:
Accuracy: 0.715
Decision Tree Accuracy: 0.655
