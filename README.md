# Ex-06-Feature-Transformation
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM: 
### STEP 1: Read the given Data
### STEP 2: Clean the Data Set using Data Cleaning Process
### STEP 3: Apply Feature Transformation techniques to all the features of the data set
### STEP 4: Print the transformed features

Developed by: NATHIN R

Register No. : 212222230090

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)

df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()
```
## Output:
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/8d64e534-7eed-4fd6-b6d0-47d5ff2c2208)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/9a14b8f2-452a-4d86-b58d-36d75adc68ee)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/a5e2a257-2c3c-4f89-83e3-e5cb101441de)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/91dcaa1f-bb4d-4ccb-b5ce-5d35cf0d1111)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/1b93eee0-6bb1-47ab-8a4f-d1e72c9289b2)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/7c28d359-f45d-42a1-b58f-50c132fcd423)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/e3b42334-2a27-4241-86c4-83ee7797033e)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/2b992f11-f5af-407a-bb3f-20565e0c8761)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/e48a63af-60c1-41c1-a8d2-4c995f9a634d)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/6ec60478-03cf-43b4-9204-a713633b8e50)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/5e9bc590-cddd-41fc-acd2-1641eeaba639)
![image](https://github.com/NathinR/Ex-06-Feature-Transformation/assets/118679646/40b11a8e-012a-42a7-9efd-d978cb554a53)

## RESULT:
Thus Feature transformation is performed and executed successfully for the given dataset
