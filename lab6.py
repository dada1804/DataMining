 import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('credit-g_csv.csv')

# Selecting desired attributes
selected_attributes = ['duration', 'credit_history', 'credit_amount','employment','other_parties', 'existing_credits', 'class']  # Modify this list with the desired attribute names

# Selecting columns based on the desired attributes
dataset_selected = dataset[selected_attributes]

# Performing label encoding for categorical attributes
label_encoder = LabelEncoder()
for column in dataset_selected.columns:
    if dataset_selected[column].dtype == 'object':
        dataset_selected[column] = label_encoder.fit_transform(dataset_selected[column].astype(str))

# Splitting the dataset into input features (X) and target variable (y)
X = dataset_selected.drop('class', axis=1)
y = dataset_selected['class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Decision Tree model
model = DecisionTreeClassifier()

# Training the Decision Tree model
model.fit(X_train, y_train)

# Evaluating the model on the test set
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)


Output:
Accuracy: 0.615

