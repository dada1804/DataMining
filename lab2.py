import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
# Assuming you have the dataset stored in a CSV file named 'credit-g_csv.csv'
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
# Creating the OneR classifier
oneR = DecisionTreeClassifier(max_depth=6)
oneR.fit(X_train, y_train)
# Get the feature importances (information gain values)
importance = oneR.feature_importances_
# Create a dictionary to store the attribute and its information gain value
attribute_ig = {}
for i, feature in enumerate(X.columns):
attribute_ig[feature] = importance[i]
# Sort the information gain values in descending order
sorted_ig = sorted(attribute_ig.items(), key=lambda x: x[1], reverse=True)
# Display the sorted information gain values
for attribute, ig in sorted_ig:
print(f"Information Gain for {attribute}: {ig}")


Output:
Information Gain for checking_status: 0.2320764755779459
Information Gain for credit_amount: 0.13759829875865817
Information Gain for duration: 0.09321097756053262
Information Gain for credit_history: 0.0886973215596801
Information Gain for purpose: 0.07480056563862388
Information Gain for other_parties: 0.07239771581170666
Information Gain for age: 0.06793266679615631
Information Gain for savings_status: 0.05548364930771497
Information Gain for residence_since: 0.044748464323232276
Information Gain for other_payment_plans: 0.026578924926822588
Information Gain for personal_status: 0.023192544903901477
Information Gain for existing_credits: 0.021465897537311398
Information Gain for num_dependents: 0.01789062819892768
Information Gain for job: 0.01431250255914215
Information Gain for property_magnitude: 0.014293520364589625
Information Gain for own_telephone: 0.010591498590160921
Information Gain for employment: 0.004728347584893269
Information Gain for installment_commitment: 0.0
Information Gain for housing: 0.0
Information Gain for foreign_worker: 0.0
Take 5 or 6 attributes which has higher IG value.
