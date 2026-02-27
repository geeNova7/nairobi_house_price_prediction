import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/nairobi_property_prices.csv')

print(df.head())
print(df.info())
print(df.describe())

print(df.isnull().sum())

#price distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Price'],kde=True)
plt.title('distribution of house prices')
plt.show()

#Location vs price
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Location'],y=df['Price'])
plt.title('Location vs Price')
plt.show()

#correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues')
plt.title('Feature Correletion')
plt.show()