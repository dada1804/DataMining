import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.preprocessing import LabelEncoder

dataset=pd.read_csv('credit-g_csv.csv')

label_encoder=LabelEncoder()
for column in dataset.columns:
    if dataset[column].dtype=='object':
        dataset[column]=label_encoder.fit_transform(dataset[column])

X=dataset.drop('class',axis=1)
y=dataset['class']

model=DecisionTreeClassifier()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
cross=cross_val_score(model,X,y,cv=5)
model.fit(X_train,y_train)
treeacc=model.score(X_test,y_test)
print("DecisionTree accuracy:",treeacc)
print("Cross val accuracy:",cross)
print("Avg of cross val accuracy:",cross.mean())
