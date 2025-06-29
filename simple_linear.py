import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5])
y = np.array([30,35,40,45,50])

mean_x = np.mean(x)
mean_y = np.mean(y)


numerator = 0
denominator = 0



for i in range(len(x)):
  numerator += (x[i]-mean_x)*(y[i]-mean_y)
  denominator += (x[i]-mean_x)**2

b1 = numerator/denominator #cov(x,y)/var(x)
b0 = mean_y-b1*mean_x



print(f"Intercept(b0):{b0}")
print(f"Slope (b1): {b1}")


y_pred = b0+b1*x
print(y_pred)



inp_x = float(input("Enter a value of  x:"))
op_y = b0+b1*inp_x
print(f"Predicted Y for input {inp_x}:{op_y}")

plt.scatter(x,y,color='blue',label='Actual Data')
plt.plot(x,y_pred,color='red',label='Regression line')
plt.scatter(inp_x,op_y,color='green',marker='x',label='Predicted Data')
plt.legend()
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Linear Regression Example")
plt.show()