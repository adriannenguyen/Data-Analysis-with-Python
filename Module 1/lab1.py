# import pandas library
import pandas as pd
import numpy as np

# Read the online file by the URL provides above, and assign it to variable "df"
other_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"
df = pd.read_csv(other_path, header=None)

# Check the bottom 10 rows of data frame "df"
print(df.tail(10))

# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)

# replace headers and recheck dataframe
df.columns = headers
print(df.head(10))

# We need to replace the "?" symbol with NaN so the dropna() can remove the missing values:
df1 = df.replace('?',np.NaN)

# We can drop missing values along the column "price" as follows:
df = df1.dropna(subset=["price"], axis=0)
print(df.head(20))

# Find the name of the columns of the dataframe.
print(df.columns)

# Apply the method to ".describe()" to the columns 'length' and 'compression-ratio'.
print(df[["length", "compression-ratio"]].describe())