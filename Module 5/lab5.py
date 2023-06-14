import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'

df = pd.read_csv(path)

df.to_csv('module_5_auto.csv')

# First, let's only use numeric data:
df=df._get_numeric_data()
df.head()

# Libraries for plotting:
from ipywidgets import interact, interactive, fixed, interact_manual

# Functions for Plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

# An important step in testing your model is to split your data into training and testing data. 
# We will place the target data price in a separate dataframe y_data:
y_data = df['price']

# Drop price data in dataframe x_data:
x_data=df.drop('price',axis=1)

# Now, we randomly split our data into training and testing data using the function train_test_split.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# Use the function "train_test_split" to split up the dataset such that 40% of the data samples will be utilized for testing. 
# Set the parameter "random_state" equal to zero. 
# The output of the function should be the following: "x_train1" , "x_test1", "y_train1" and "y_test1"

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.40, random_state=0)

print("number of test samples :", x_test1.shape[0])
print("number of training samples:",x_train1.shape[0])

# Let's import LinearRegression from the module linear_model.
from sklearn.linear_model import LinearRegression

# We create a Linear Regression object:
lre=LinearRegression()

# We fit the model using the feature "horsepower":
lre.fit(x_train[['horsepower']], y_train)

# Let's calculate the R^2 on the test data:
lre.score(x_test[['horsepower']], y_test)

# We can see the R^2 is much smaller using the test data compared to the training data.
lre.score(x_train[['horsepower']], y_train)

# Find the R^2 on the test data using 40% of the dataset for testing.
lre.fit(x_train1[["horsepower"]],y_train1)
lre.score(x_train1[["horsepower"]], y_test1)

# Let's import model_selection from the module cross_val_score.
from sklearn.model_selection import cross_val_score

# We input the object, the feature ("horsepower"), and the target data (y_data). 
# The parameter 'cv' determines the number of folds. In this case, it is 4.
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

# The default scoring is R^2. Each element in the array has the average R^2 value for the fold:
Rcross

# We can calculate the average and standard deviation of our estimate:
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

# We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')

# Calculate the average R^2 using two folds, then find the average R^2 for the second fold utilizing the "horsepower" feature:
Rcross2 = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)
print(Rcross2.mean())

# You can also use the function 'cross_val_predict' to predict the output. 
# The function splits up the data into the specified number of folds, 
# with one fold for testing and the other folds are used for training. First, import the function:
from sklearn.model_selection import cross_val_predict

# We input the object, the feature "horsepower", and the target data y_data. 
# The parameter 'cv' determines the number of folds. In this case, it is 4. We can produce an output:
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]

# Let's create Multiple Linear Regression objects and train the model using 
# 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg' as features
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# Prediction using training data:
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]

# Prediction using test data:
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]

# Let's examine the distribution of the predicted values of the training data.
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

# When the model generates new values from the test data, 
# we see the distribution of the predicted values is much different from the actual target values.
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)

#  Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset
from sklearn.preprocessing import PolynomialFeatures

