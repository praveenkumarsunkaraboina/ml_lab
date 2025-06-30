import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

x = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,1,1,1,1,1,1])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)

y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

print(f"Accuracy of the model: {accuracy*100:.2f}%")

print("Confusion Matrix:\n",confusion_matrix(y_pred,y_test))

print("Classification Report:\n",classification_report(y_test,y_pred))

new_data = np.array([[4.63525],[11],[0]])

new_pred = log_reg.predict(new_data)

for i,val in enumerate(new_data):
    print(f'Hours Studied: {val[0]}, Prediction: {"Failed" if new_pred[i]==0 else "Passed"}')

plt.scatter(x,y,label='Data Points')
plt.scatter(new_data,new_pred,label='Predicted',marker='*')
plt.plot(x,log_reg.predict(x),label='Logistic Regression')

plt.xlabel('Hours Studied')
plt.ylabel('Passed(1)/Failed(0)')
plt.title('Logistic Regression: Hours Studied vs Pass/Fail')
plt.legend()
plt.show()