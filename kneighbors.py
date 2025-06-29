from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_pred,y_test)

print(f"Model Accuracy: {acc:.2f}")


plt.figure(figsize=(8,6))
plt.scatter(X_test[:,0],X_test[:,1],c=y_pred)
plt.title('KNN Predicted Classes')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()

print(f"\nLet's Classify a new Iris flower!")
user_data = []
for feature_name in iris.feature_names:
    val = float(input(f"Enter {feature_name}:"))
    user_data.append(val)

user_data_scaled = scaler.transform([user_data])

pred_class = model.predict(user_data_scaled)
predicted_name = iris.target_names[pred_class[0]]
print(f"The model predicts this flower as:",str(predicted_name))



