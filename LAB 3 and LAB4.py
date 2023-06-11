import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


#Assuming you have the dataset stored in a CSV file named 'dataset.csv'
dataset = pd.read_csv('dataset.csv')

#Encoding categorical variables using label encoding
label_encoder = LabelEncoder()
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = label_encoder.fit_transform(dataset[column])

# Splitting the dataset into input features (X) and target variable (y)
X = dataset.drop('class', axis=1)  # Replace 'credit_approval' with the actual target column name
y = dataset['class']  # Replace 'credit_approval' with the actual target column name

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Decision Tree model
model = DecisionTreeClassifier()

# Training the Decision Tree model
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Calculating the accuracy
accuracy=accuracy_score(y_test,y_pred)
print("Decision Tree Model:")
print(model)
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=list(X.columns), class_names=['0', '1'], filled=True)
print("\nEvaluation Metrics:")
print("Accuracy:", accuracy)
perc_crct=(accuracy*100)
perc_wrong=100-perc_crct
print("Correctly classified :",perc_crct,"%")
print('Wrongly Classified: ',perc_wrong,"%")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Reporting the model and evaluation metrics
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", mse ** 0.5)


Output:
	Decision Tree Model:
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')

Evaluation Metrics:
Accuracy: 0.67
Correctly classified : 67.0 %
Wrongly Classified:  33.0 %

Mean Absolute Error: 0.32
Root Mean Squared Error: 0.565685424949238
