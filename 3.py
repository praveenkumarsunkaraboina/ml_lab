# Pandas
import pandas as pd

data = {
    'Name':['Praveen','Akhil','Sai Pavan','Ashwin'],
    'Age':[21,21,20,20],
    'City':['Nalgonda','Hyderabad','Khammam','Hyderabad']
}

df = pd.DataFrame(data)
print(df)

df.head()

df.info()

df.describe()

print(df['Name']) # select a column

print(df[df['Age']>20]) # Filtering rows

df.fillna(value=0,inplace=True)

df.dropna(inplace=True)

df.isnull().sum()

grouped = df.groupby('City')['Age'].mean()
print(grouped)

df2 = pd.DataFrame({'City':['Nalgonda','Khammam','Hyderabad'],'Population':[8_000_000,6_000_000,20_000_000]})
merged_df = pd.merge(df,df2,on='City')
print(merged_df)


# matplotlib

import matplotlib.pyplot as plt

x = [1,2,3,4,5]
y = [10,20,30,40,50]

plt.plot(x,y)
plt.title('Line Plot Example')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()

#scatter plot
plt.scatter(x,y)
plt.title("Scatter Plot Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

data = [1,2,2,3,3,3,4,5,5,6]
plt.hist(data,bins=5,color='blue',edgecolor='black')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


categories = ['A','B','C','D']
values = [5,7,3,4]
plt.bar(categories,values)
plt.title("Bar Chart")
plt.xlabel("Category")
plt.ylabel("Value")
plt.show()

plt.pie(values,labels=categories,explode=(0,0.1,0,0),autopct='%1.1f%%',shadow=True,startangle=140)
plt.title("Pie Chart")
plt.axis("equal")
plt.show()