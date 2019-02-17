# Simple Linear Regession

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')

# Create matrix of features. Taking all cols -1 values
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)

# Visualising the Training set results
# 1. Plotting using scatter graph
plt.scatter(X_train, y_train, color = 'red') 
# 2. Here plot X_train data vs Linear Regression Predicted X_train data just for 
#    understanding Accuracy of our prediction model
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# Setting the title for Graph
plt.title('Salary vs Experience(Training set)')
# Setting label for X and Y axis
plt.xlabel('Experience in years')
plt.ylabel('Salary')
# Finally show the graph
plt.show()

# Visualising the Test set results
# 1. Plotting using scatter graph
plt.scatter(X_test, y_test, color = 'red') 
# 2. Here plot X_train data vs Linear Regression Predicted X_train data just for 
#    understanding Accuracy of our prediction model
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# Setting the title for Graph
plt.title('Salary vs Experience(Test set)')
# Setting label for X and Y axis
plt.xlabel('Experience in years')
plt.ylabel('Salary')
# Finally show the graph
plt.show()
