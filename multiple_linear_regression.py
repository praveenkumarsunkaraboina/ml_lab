import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

housing = fetch_california_housing()

X = housing.data
y = housing.target

df = pd.DataFrame(X,columns=housing.feature_names)
df['target']=y
df.head()

X_train, X_test, Y_train, Y_test = train_test_split(X,y,random_state=42,test_size=0.2)

model = LinearRegression()

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test,y_pred)

r2 = r2_score(Y_test,y_pred)

print("Mean Squared Error:",mse)
print("R2_Score:",r2)




plt.figure(figsize=(5,5))
y_pred = model.predict(X_test)
plt.scatter(Y_test,y_pred,alpha=0.5,label='Actual Vs Predicted')
plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()],color='red',label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.show()



x=[8.3252,41.0,6.984127,1.023810,322.0,2.555556,37.88,-122.23]
y_pred = model.predict([x])
print("Prediction:",y_pred)
