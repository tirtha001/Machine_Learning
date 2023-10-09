import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print(x)
# print(y)

# As there is no missing data, we don't need to work with it
# Also there is no categorical data. All we have - Numerical data

# Now we can move to train-test split
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= .2, random_state= 1)
# print("Train Data for X: \n", x_train)
# print("Test Data for X: \n ",x_test)
# print("Train Data for Y: \n ",y_train)
# print("Test Data for Y: \n ",y_test)

# Training the Simple Linear Regression model on the Training Set
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
# Now we've to train it. So we need to somehow connect it with the training set.
# The action or function we'd need in order to connect it -> fit function
lin_reg.fit(x_train, y_train)

# Predict the test result based on the trained model
lin_reg.predict(x_test)

# Let's Visualize our model tasks on plotting graphs
# Visualize the training set result
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, lin_reg.predict(x_train), color = "blue")
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualize the test set result
plt.scatter(x_test, y_test, color = "Yellow")
plt.plot(x_train, lin_reg.predict(x_train), color = "black")
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
