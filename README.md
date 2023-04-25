# Ex-06-Feature-Transformation
## AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM:
STEP 1:
Read the given Data

STEP 2:
Clean the Data Set using Data Cleaning Process

STEP 3:
Apply Feature Transformation techniques to all the features of the data set

STEP 4:
Save the data to the file

## PROGRAM:
```python
# Developed by Udayakumar R
# Ref. No: 22008609

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import statsmodels.api as sm
import scipy.stats as stats
df=pd.read_csv("Data_to_Transform.csv")
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()
```
## OUTPUT:

![image](https://user-images.githubusercontent.com/118708024/234162755-324d0144-1991-4cb7-b551-b1363f015c94.png)
![image](https://user-images.githubusercontent.com/118708024/234162863-10a11ee2-8104-436c-9302-da0fc1c4dfe5.png)
![image](https://user-images.githubusercontent.com/118708024/234162978-b39d7dc5-6cd5-45d1-982d-49faee15c5a4.png)
![image](https://user-images.githubusercontent.com/118708024/234163226-42bb6f50-ee94-4ae3-80c3-b055e5a58733.png)
![image](https://user-images.githubusercontent.com/118708024/234163296-493e2cb7-1cb3-4043-9b19-def4ce9796c0.png)
![image](https://user-images.githubusercontent.com/118708024/234163355-ee121f08-4231-48b1-8026-6f2c5464d44b.png)
![image](https://user-images.githubusercontent.com/118708024/234163417-653ae236-7645-4cf2-9a30-4a2247ecf979.png)
![image](https://user-images.githubusercontent.com/118708024/234163506-db023832-6e72-4fd9-8ad7-498d3a558b48.png)
![image](https://user-images.githubusercontent.com/118708024/234163554-8c95b442-6b36-4ecc-af85-dec53289bdb3.png)
![image](https://user-images.githubusercontent.com/118708024/234163597-048d99b7-a1ec-4a25-a41c-1e8e01ef7a43.png)

## RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully.
