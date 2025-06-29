import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score


iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target

label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save the encoder

scaler = StandardScaler()

X = scaler.fit_transform(df.select_dtypes(include=['float64','int64']).values)

X_train,_ = train_test_split(X,test_size=0.2,random_state=42)

kmeans = KMeans(n_clusters=3,random_state=42).fit(X_train)

silhouette_score = silhouette_score(X_train,kmeans.labels_)

print(f"Silhouette Score: {silhouette_score}")

new_data_point = {
    'sepal length (cm)':2.5,
    'sepal width (cm)':3.5,
    'petal length (cm)':4.5,
    'petal width (cm)':2.5,
    'target':0
}

new_df = pd.DataFrame([new_data_point])

# Apply the saved label encoders
for col, le in label_encoders.items():
    if col in new_df:
        new_df[col] = le.transform(new_df[col])


new_data_point_scaled = scaler.transform(new_df.values)

predicted_cluster = kmeans.predict(new_data_point_scaled)

print("Predicted Cluster:",iris.target_names[predicted_cluster[0]])


plt.figure(figsize=(8,6))
plt.scatter(X_train[:,0],X_train[:,1],c=kmeans.labels_)
plt.scatter(new_data_point_scaled[:,0],new_data_point_scaled[:,1],color='green',marker='*',label=str(iris.target_names[predicted_cluster[0]]))
for i, center in enumerate(kmeans.cluster_centers_):
    plt.scatter(center[0],center[1],marker='X',label=str(iris.target_names[i]))


plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('K-Means Clustering')
plt.show()
