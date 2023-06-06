import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
# Assuming you have the dataset stored in a CSV file named 'dataset.csv'
dataset = pd.read_csv('credit-g_csv.csv')
# Performing label encoding for categorical attributes
label_encoder = LabelEncoder()
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = label_encoder.fit_transform(dataset[column])
# Splitting the dataset into input features (X) and target variable (y)
X = dataset.drop('class', axis=1)
y = dataset['class']
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Creating the OneR classifier
oneR = DecisionTreeClassifier(max_depth=1)
# Training the OneR classifier
oneR.fit(X_train, y_train)
# Predicting the target variable using the trained OneR classifier
y_pred = oneR.predict(X_test)
# Calculating the accuracy of the OneR classifier
accuracy = accuracy_score(y_test, y_pred)
# Extracting the chosen attribute by the OneR classifier
chosen_attribute = X.columns[oneR.tree_.feature[0]]
# Print the results
print("Chosen Attribute by OneR:", chosen_attribute)
print("Accuracy of OneR Classifier:", accuracy)

Output:
	Chosen Attribute by OneR: checking_status
Accuracy of OneR Classifier: 0.705
