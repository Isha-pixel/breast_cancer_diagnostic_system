#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
df = pd.read_csv('data.csv')
print(df.head(7))

#Count the number of rows and columns in the data set
print(df.shape)

#Count the empty (NaN, NAN, na) values in each column
print(df.isna().sum())

#Drop the column with all missing values (na, NAN, NaN)
#This drops the column 'Unnamed'
df = df.dropna(axis=1)

#Get the new count of the number of rows and cols
print(df.shape)

#Get a count of the number of 'M' & 'B' cells
print(df['diagnosis'].value_counts())

#Visualize this count
sns.countplot(x=df['diagnosis'],label="Count")

# Look at the data types
print(df.dtypes)

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))

print(df.head(5))

#Generate the pairplot
sns.pairplot(df.iloc[:,1:10], hue="diagnosis")

#Get the correlation of the columns
print(df.iloc[:,1:33].corr())

#Dropping columns with correlation greater than 0.95
cor_matrix = df.corr().abs()
print(cor_matrix)

#Selecting upper triangular matrix
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
print(upper_tri)

#Drop the columns with high correlation
to_drop = [i-1 for i,column in enumerate(upper_tri.columns) if any(upper_tri[column] > 0.95)]
print();
print("Columns dropped ",to_drop)

df = df.drop(columns=df.columns[to_drop], axis=1)
print();
print(df.head())

#Generating heatmaps
plt.figure(figsize=(15,15))
sns.heatmap(df.iloc[:,1:25].corr(), annot=True, fmt='.0%')
plt.show()