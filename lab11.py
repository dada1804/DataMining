import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load the credit dataset
credit_data = pd.read_csv('credit-g_csv.csv')
# Preprocess the data
numeric_features = ['age', 'duration', 'credit_amount'] # Example of numeric features
categorical_features = ['other_parties', 'property_magnitude'] # Example of categorical
# Apply label encoding to categorical features
label_encoder = LabelEncoder()
for feature in categorical_features:
    credit_data[feature] = label_encoder.fit_transform(credit_data[feature])
# Combine numeric and encoded categorical features
features = numeric_features + categorical_features
X = credit_data[features]
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Perform k-means clustering
k = 3 # Number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
# Get the cluster labels
labels = kmeans.labels_
# Add the cluster labels to the dataset
credit_data['Cluster'] = labels
# Print the cluster assignments
print(credit_data['Cluster'].value_counts())
