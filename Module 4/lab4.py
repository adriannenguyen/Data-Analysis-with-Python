import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork
import warnings
from sklearn.linear_model import LinearRegression
warnings.filterwarnings('ignore')

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

df = pd.read_csv(filename, header=0)

# Create the linear regression object
lm = LinearRegression()
print(lm)

X = df[['highway-mpg']]
Y = df['price']

# Fit the linear model using highway-mpg:
lm.fit(X,Y)

# We can output a prediction:
Yhat=lm.predict(X)
print(Yhat[0:5])

# What is the value of the intercept (a)?
print(lm.intercept_)

# What is the value of the slope (b)?
print(lm.coef_)

# Create a linear regression object called "lm1".
lm1 = LinearRegression()

# Train the model using "engine-size" as the independent variable and "price" as the dependent variable?
lm1.fit(df[["engine-size"]], df[["price"]])

# Find the slope and intercept of the model.
print(lm1.coef_)
print(lm1.intercept_)

# What is the equation of the predicted line? You can use x and yhat or "engine-size" or "price".
Yhat = -7963.34 + 166.86 * X
Price= -7963.34 + 166.86 * df['engine-size']

# Let's develop a model using these variables as the predictor variables.
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# Fit the linear model using the four above-mentioned variables.
lm.fit(Z, df['price'])

# What is the value of the intercept(a)?
print(lm.intercept_)

# What are the values of the coefficients (b1, b2, b3, b4)?
print(lm.coef_)

# Create and train a Multiple Linear Regression model "lm2" where the response variable is "price", 
# and the predictor variable is "normalized-losses" and "highway-mpg".
lm2 = LinearRegression()
lm2.fit(df[["normalized-losses", "highway-mpg"]], df["price"])

#Find the coefficient of the model.
print(lm2.coef_)

# Let's visualize highway-mpg as potential predictor variable of price:
import seaborn as sns

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

# Let's compare this plot to the regression plot of "peak-rpm".
plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

# Given the regression plots above, is "peak-rpm" or "highway-mpg" 
# more strongly correlated with "price"? 
# Use the method ".corr()" to verify your answer.
df[["peak-rpm","highway-mpg","price"]].corr()

# Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'],y=df['price'])
plt.show()

# First, let's make a prediction:
Y_hat = lm.predict(Z)

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()

# We will use the following function to plot the data:
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

# Let's get the variables:
x = df['highway-mpg']
y = df['price']

# Let's fit the polynomial using the function polyfit, then use the function poly1d to display the polynomial function.
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)

# Let's plot the function:
PlotPolly(p, x, y, 'highway-mpg')
np.polyfit(x, y, 3)

# Create 11 order polynomial model with the variables x and y from above.
f1 = np.polyfit(x, y, 11)
p1 = np.poly1d(f1)
print(p1)
PlotPolly(p1, x, y, "Highway MPG")

# We can perform a polynomial transform on multiple features. First, we import the module:
from sklearn.preprocessing import PolynomialFeatures

We create a PolynomialFeatures object of degree 2:
pr=PolynomialFeatures(degree=2)
pr

Z_pr=pr.fit_transform(Z)

# In the original data, there are 201 samples and 4 features.
Z.shape

# After the transformation, there are 201 samples and 15 features.
Z_pr.shape

# Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. 
# We also use StandardScaler as a step in our pipeline.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# We create the pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constructor.
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# We input the list as an argument to the pipeline constructor:
pipe=Pipeline(Input)

# First, we convert the data type Z to type float to avoid conversion warnings that may appear as a result of StandardScaler taking float inputs.
# Then, we can normalize the data, perform a transform and fit the model simultaneously.
Z = Z.astype(float)
pipe.fit(Z,y)

# Similarly, we can normalize the data, perform a transform and produce a prediction simultaneously.
ypipe=pipe.predict(Z)
ypipe[0:4]

# Create a pipeline that standardizes the data, then produce a prediction using a linear regression model using the features Z and target y.
Input=[('scale',StandardScaler()),('model',LinearRegression())]

pipe=Pipeline(Input)

pipe.fit(Z,y)

ypipe=pipe.predict(Z)
ypipe[0:10]

# Model 1: Simple Linear Regression
# Let's calculate the R^2:
#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))

# Let's calculate the MSE:
# We can predict the output i.e., "yhat" using the predict method, where X is the input variable:
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])

# Let's import the function mean_squared_error from the module metrics:
from sklearn.metrics import mean_squared_error

# We can compare the predicted results with the actual results:
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

# Model 2: Multiple Linear Regression
# Let's calculate the R^2:
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))

# Let's calculate the MSE.
# We produce a prediction:
Y_predict_multifit = lm.predict(Z)

# We compare the predicted results with the actual results:
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

# Model 3: Polynomial Fit
# Let's calculate the R^2.
# Let’s import the function r2_score from the module metrics as we are using a different function.
from sklearn.metrics import r2_score

# We apply the function to get the value of R^2:
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

# We can also calculate the MSE:
mean_squared_error(df['price'], p(x))

# In the previous section, we trained the model using the method fit. 
# Now we will use the method predict to produce a prediction. 
# Lets import pyplot for plotting; we will also be using some functions from numpy.
import matplotlib.pyplot as plt
import numpy as np

# Create a new input:
new_input=np.arange(1, 100, 1).reshape(-1, 1)

# Fit the model:
lm.fit(X, Y)

# Produce a prediction:
yhat=lm.predict(new_input)
yhat[0:5]

# We can plot the data:
plt.plot(new_input, yhat)
plt.show()
