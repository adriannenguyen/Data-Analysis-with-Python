import pandas as pd
import numpy as np
import skillsnetwork
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"

df = pd.read_csv(filename, header=0)

# What is the data type of the column "peak-rpm"?
print(df["peak-rpm"].dtypes)

# Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower.
print(df[["bore","stroke","compression-ratio","horsepower"]].corr())

# Engine size as potential predictor variable of price
# sns.regplot(x="engine-size", y="price", data=df)
# plt.ylim(0,)

# We can examine the correlation between 'engine-size' and 'price' and see that it's approximately 0.87.
print(df[["engine-size", "price"]].corr())

# Let's find the scatterplot of "highway-mpg" and "price".
# sns.regplot(x="highway-mpg", y="price", data=df)

# We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704
print(df[['highway-mpg', 'price']].corr())

# Let's see if "peak-rpm" is a predictor variable of "price".
# sns.regplot(x="peak-rpm", y="price", data=df)

# We can examine the correlation between 'peak-rpm' and 'price' and see it's approximately -0.101616.
print(df[['peak-rpm','price']].corr())

# Find the correlation between x="stroke" and y="price".
print(df[['stroke','price']].corr())

# Given the correlation results between "price" and "stroke", do you expect a linear relationship?
# sns.regplot(x="stroke", y="price", data=df)

# Let's look at the relationship between "body-style" and "price".
# sns.boxplot(x="body-style", y="price", data=df)

# Let's examine engine "engine-location" and "price":
sns.boxplot(x="engine-location", y="price", data=df)

# Let's examine "drive-wheels" and "price".
sns.boxplot(x="drive-wheels", y="price", data=df)

# We can apply the method "describe" as follows:
df.describe()

# We can apply the method "describe" on the variables of type 'object' as follows:
print(df.describe(include=['object']))

#  We can apply the "value_counts" method on the column "drive-wheels". 
# Don’t forget the method "value_counts" only works on pandas series, not pandas dataframes
df['drive-wheels'].value_counts()

# We can convert the series to a dataframe as follows:
df['drive-wheels'].value_counts().to_frame()

# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
print(drive_wheels_counts)

# Now let's rename the index to 'drive-wheels':
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

# We can repeat the above process for the variable 'engine-location'.
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts.head(10))

# let's group by the variable "drive-wheels"
df['drive-wheels'].unique()

# We can select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".
df_group_one = df[['drive-wheels','body-style','price']]

# We can then calculate the average price for each of the different categories of data.
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()

# let's group by both 'drive-wheels' and 'body-style
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()

# we will leave the drive-wheels variable as the rows of the table, and pivot body-style to become the columns of the table:
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')

# We can fill these missing cells with the value 0, but any other value could potentially be used as well
grouped_pivot = grouped_pivot.fillna(0)

# Use the "groupby" function to find the average "price" of each car based on "body-style".
df_group_two = df[["price", "body-style"]]
df_group_two = df_group_two.groupby(["body-style"], as_index = False).mean()

# Let's use a heat map to visualize the relationship between Body Style vs Price.
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

# The default labels convey no useful information to us. Let's change that:
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

# Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Let's calculate the Pearson Correlation Coefficient and P-value of 'horsepower' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value) 

# Let's calculate the Pearson Correlation Coefficient and P-value of 'length' and 'price'.
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

# Let's calculate the Pearson Correlation Coefficient and P-value of 'width' and 'price':
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value ) 

# Let's calculate the Pearson Correlation Coefficient and P-value of 'curb-weight' and 'price':
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)

# Let's calculate the Pearson Correlation Coefficient and P-value of 'engine-size' and 'price':
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 

# Let's calculate the Pearson Correlation Coefficient and P-value of 'bore' and 'price':
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =  ", p_value ) 

# City-mpg vs. Price
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

# Highway-mpg vs. Price
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print( "The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value ) 

# To see if different types of 'drive-wheels' impact 'price', we group the data.
grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)

# We can obtain the values of the method group using the method "get_group".
grouped_test2.get_group('4wd')['price']

# We can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   

# fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )

# 4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)  

# 4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val)  

