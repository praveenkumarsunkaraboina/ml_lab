import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd

data = {
    'X': [1.0, 1.5, 5.0, 8.0, 1.0, 9.0, 8.0, 10.0, 9.0],
    'Y': [2.0, 1.8, 8.0, 8.0, 0.6, 11.0, 2.0, 2.0, 3.0]
}

df = pd.DataFrame(data)

df.to_csv('sample.csv',index=False)

def load_dataset():
    file_path = input("Enter CSV file path:").strip()
    df=pd.read_csv(file_path)
    return df

def perform_kmeans(data,k):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=k,random_state=0)
    kmeans.fit(scaled_data)
    data['Cluster']=kmeans.labels_
    return data, scaled_data, kmeans.labels_

def plot_clusters(data,labels):
    plt.scatter(data=data,x=data.columns[0],y=data.columns[1],c=labels)
    plt.title("K-Means Clustering")
    plt.show()

def main():
    print("=== Simple K-Means Clustering with Silhouette Score ===")
    data = load_dataset()
    k = int(input("Enter number of clusters (K): "))
    clustered_data, scaled_data, labels = perform_kmeans(data.copy(), k)
    score = silhouette_score(scaled_data, labels)
    print(f"\nSilhouette Score for k={k}: {score:.4f}")
    plot_clusters(clustered_data,labels)

main()