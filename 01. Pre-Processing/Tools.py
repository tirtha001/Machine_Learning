from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Dataset Import
dataset = pd.read_csv("Data.csv")  # Creates a data-frame of the dataset
# ":"-> means range. 1st ':' covers all the rows. 2nd ':' covers the cols range. '-1' means last column
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# print("Value of x \n: ", x)
# print("Value of y \n: ", y)

# Taking care of missing data (Numerical Data)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# we need to apply fit-method to connect with the matrix of features and also replace the missing fields
imputer.fit(x[:, 1:3])
# transform to update the value
x[:, 1:3] = imputer.transform(x[:, 1:3])
# print("Updated Feature Matrix: \n", x)

# Encoding Categorical Data
# Encoding Independent Variable
colTrans = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(colTrans.fit_transform(x))

# print("Numerical Transformed matrix: \n", x)

# Encode the dependent variable-> Label encoding that form 0 & 1
depen = LabelEncoder()
y = depen.fit_transform(y)
# print("Transform the dependent column: \n", y)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

# print("Train Data of X: \n", x_train)
# print("Test Data of X: \n", x_test)
# print("Train Data of Y: \n", y_train)
# print("Test Data of Y: \n", y_test)

# Feature Scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print("After Feature Scaling: \n")
print("Train Matrix = \n", x_train)
print("Test of X= \n", x_test)
