import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Dataset Loading
df = pd.read_csv('dataset.csv')

print("Dataset Preview:")
print(df.head())

print("\nDataset Summary:")
print(df.describe().T)  

df_numeric = df.select_dtypes(include=[np.number])

print("Mean Values:\n", df_numeric.mean())
print("Median Values:\n", df_numeric.median())
print("Mode Values:\n", df_numeric.mode().iloc[0])
print("Standard Deviation:\n", df_numeric.std())
print("Variance:\n", df_numeric.var())
print("Skewness:\n", df_numeric.skew())
print("Kurtosis:\n", df_numeric.kurt())


# Step 2: Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("\nMissing Values:")
print(df.isnull().sum())

for col in df.columns:
    if df[col].dtype == 'object':  
        df[col] = df[col].fillna(df[col].mode()[0])
    else:  
        df[col] = df[col].fillna(df[col].mean()) 

print("\nCorrelation Matrix:")
correlation_matrix = df_numeric.corr()
print(correlation_matrix)

plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

print("\nCovariance Matrix:")
covariance_matrix = df_numeric.cov()
print(covariance_matrix)

target_column = 'Outcome'  # Change target
if target_column not in df.columns:
    raise ValueError(f"Column '{target_column}' not found in dataset!")

X = df.drop(columns=[target_column])
y = df[target_column]

label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  
    label_encoders[col] = le  

df_numeric.hist(figsize=(12, 8), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 3: Train a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("The Decision Tree model achieved an accuracy of", round(accuracy_score(y_test, y_pred) * 100, 2), "%.")


# Step 5: Elbow Method KMEANS


from sklearn.cluster import KMeans

wcss = []  
for i in range(1, 11):  
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_numeric)  
    wcss.append(kmeans.inertia_) 

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.show()


# Step 6: K-Means Clustering

X_numeric = df.select_dtypes(include=[np.number])

kmeans = KMeans(n_clusters=2, random_state=42)

df['Cluster'] = kmeans.fit_predict(X_numeric)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_numeric.iloc[:, 0], y=X_numeric.iloc[:, 1], hue=df['Cluster'], palette="viridis")
plt.title("K-Means Clustering")
plt.xlabel(X_numeric.columns[0])
plt.ylabel(X_numeric.columns[1])
plt.legend(title='Cluster', loc='best')
plt.show()

print("\nFirst few rows with Clusters:")
print(df.head())

print("\nCluster Centers:")
print(kmeans.cluster_centers_)



# Below code for Prediction

'''
# Step 3: Train a Decision Tree Model
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(criterion='squared_error', random_state=42)
model.fit(X_train, y_train)

# Step 4: Model Evaluation (Regression Metrics)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predictions
y_pred = model.predict(X_test)

# Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared Score (R²):", r2)

print(f"The Decision Tree model achieved an R-squared score of {round(r2 * 100, 2)}%.")
'''