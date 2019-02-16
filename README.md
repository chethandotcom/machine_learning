# Data Preprocessing

## Three essential libraries
1. numpy `[import numpy as np]` - Includes mathematical codes
2. matplotlib.pytplot `[import matplotlib.pyplot as plt]` - Includes plotting of Graph
3. pandas `[import pandas as pd]` - Helps in import and manage dataset

### How to handle the missing data?
1. Delete the row (Not recommended if the row holds crucial data)
2. Find the mean of the column and fill the data (Recommended)

To find the mean:
Library **SKLearn** is used to execute science equations
For example we use ScienceKit to find the mean

**Imputer definition:**
```
Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True)
```

To find the mean use the following code:
```
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

Available strategy's are:
* mean
* median
* most_frequent
```

To categorize the input data, we use **SKLearn LabelEncoder**
```
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
```

The problem with **LabelEncoder** is that _Machine Learning_ models are based on equations. Equations require numbers than strings.


Consider the following example:

Country | LabelEncoder Value
:--------:| :------------------:
France  | 0
Spain   | 1
Germany | 2

Here since `1 > 0` means Spain is greater than France, which is not the case because these countries are three different categories and not dependant variable.

To prevent *MachineLearning* model to prevent thinking like this, **Dummy Encoding** is used.

**Dummy Encoding** <br>
Here instead of having one column we will be having 3 columns like below:<br>

|Country|
|:-------:|
|France|
|Spain|
|Germany|

<br>

|France|Spain|Germany|
|:-------:|:-------:|:-------:|
|1|0|0|
|0|1|0|
|0|0|1|

We use **SKLearn OneHotEncoder** to achieve Dummy Encoding
```
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
```

**Splitting the data set into Training set and Test set**<br />
Thre reason to split the data into Training set and Test set is to avoid memoisation by model. For example consider that if we train our model with all the data we have, then model will try to remember the result for all inputs just like **by-heart**. To avoid this, it is always recommended to split the data.

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

**Feature Scaling**
When 2 variables are not in the same scale which will cause issues in Machine Learning model because most of the Machine Learning models are based on the `Euclidean Distance`

![alt text](images/euclidean_distance.png)

To scale the data, we use mostly the following

![alt text](images/feature_scaling.png)

<br>
To scale the value, we use `SKLear Preprocessing's Standard Scaler`

```
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```

<br>

### **Do we need to scale the DUMMY variables?** <br>
It depends on the context. It depends on how we want to keep interpreted in out model.<br>

Advantage | Disadvantage
:---------- | :----------
All data will be in the same scale, which is good for prediction | Will lose the interpretation of knowing which observation belongs to which

**But it wont break the model**

<br>

### **Do we need to scale dependant variable vector y_train and y_test?** <br>
NO. Since the `y_train` and `y_test` is a boolean classifier, the data avaiable is only `0 and 1`