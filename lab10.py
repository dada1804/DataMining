 # 10a If Then Rules
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Assuming you have the dataset stored in a CSV file named 'dataset.csv'
dataset = pd.read_csv('credit-g_csv.csv')

# Selecting desired attributes for the small Decision Tree
selected_attributes = ['checking_status', 'duration', 'credit_history', 'credit_amount', 'class']

# Selecting columns based on the desired attributes
dataset_selected = dataset[selected_attributes]

# Performing label encoding for categorical attributes
label_encoder = LabelEncoder()
for column in dataset_selected.columns:
    if dataset_selected[column].dtype == 'object':
        dataset_selected[column] = label_encoder.fit_transform(dataset_selected[column])

# Splitting the dataset into input features (X) and target variable (y)
X = dataset_selected.drop('class', axis=1)
y = dataset_selected['class']

# Creating the Decision Tree model with maximum depth of 3
model = DecisionTreeClassifier(max_depth=3)

# Training the Decision Tree model
model.fit(X, y)

# Converting the Decision Tree into "if-then-else" rules
def tree_to_rules(tree, feature_names, class_names):
    rules = []

    def traverse(node, rule):
        if tree.children_left[node] == -1:  # Reached a leaf node
            class_label = class_names[tree.value[node].argmax()]
            rule.append(f"THEN class = {class_label}")
            rules.append(" ".join(rule))
        else:
            feature = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            rule.append(f"IF {feature} <= {threshold}")
            traverse(tree.children_left[node], rule)
            rule[-1] = f"IF {feature} > {threshold}"
            traverse(tree.children_right[node], rule)
            rule.pop()

    traverse(0, [])

    return rules

# Define class labels for mapping
class_names = ['0', '1']

# Convert the Decision Tree into rules
decision_tree_rules = tree_to_rules(model.tree_, X.columns, class_names)

# Print the rules
for rule in decision_tree_rules:
    print(rule)

Output:
	IF checking_status <= 1.5 IF duration <= 22.5 IF credit_history <= 0.5 THEN class = 0
IF checking_status <= 1.5 IF duration <= 22.5 IF credit_history <= 0.5 IF credit_history > 0.5 THEN class = 1
IF checking_status <= 1.5 IF duration <= 22.5 IF credit_history <= 0.5 IF duration > 22.5 IF credit_amount <= 1381.5 THEN class = 0
IF checking_status <= 1.5 IF duration <= 22.5 IF credit_history <= 0.5 IF duration > 22.5 IF credit_amount <= 1381.5 IF credit_amount > 1381.5 THEN class = 0
IF checking_status <= 1.5 IF duration <= 22.5 IF credit_history <= 0.5 IF duration > 22.5 IF checking_status > 1.5 IF credit_amount <= 4158.0 IF checking_status <= 2.5 THEN class = 1
IF checking_status <= 1.5 IF duration <= 22.5 IF credit_history <= 0.5 IF duration > 22.5 IF checking_status > 1.5 IF credit_amount <= 4158.0 IF checking_status <= 2.5 IF checking_status > 2.5 THEN class = 1
IF checking_status <= 1.5 IF duration <= 22.5 IF credit_history <= 0.5 IF duration > 22.5 IF checking_status > 1.5 IF credit_amount <= 4158.0 IF checking_status <= 2.5 IF credit_amount > 4158.0 IF credit_amount <= 4241.0 THEN class = 0
IF checking_status <= 1.5 IF duration <= 22.5 IF credit_history <= 0.5 IF duration > 22.5 IF checking_status > 1.5 IF credit_amount <= 4158.0 IF checking_status <= 2.5 IF credit_amount > 4158.0 IF credit_amount <= 4241.0 IF credit_amount > 4241.0 THEN class = 1
______________________________________________________________________________________________________________________________________________________________________
# 10b OneRule
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
