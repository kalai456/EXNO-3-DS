## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv(r'D:\data science\Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/9ed547e8-77d8-491b-9ca5-6baf3a0b34f8)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/b70f645a-fc07-4ec8-b329-76c3b8e49b36)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/aff9bf1c-a5ae-40a1-b220-0389681cb074)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/91ee617f-937f-408a-b914-6431e7e07dd6)
```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

```
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/ea71712f-673b-4904-8278-0dd81b4bd66e)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/2e8adc0f-df1d-4c9e-82ab-8d43ff2e451f)
```
from category_encoders import BinaryEncoder
df=pd.read_csv(r'D:\data science\data.csv')
df
```
![image](https://github.com/user-attachments/assets/1c2d5f7e-8b57-451d-beb0-7d6b28c4c7a3)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/6e009631-9bac-4b87-b16e-7a8d66377798)
```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/c385ba6c-d83c-452e-a88e-f850d7161450)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/70c54c47-334d-40bb-8585-d77f0df151a7)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv(r'D:\data science\Data_to_Transform.csv')
df
```
![image](https://github.com/user-attachments/assets/00b813e5-9f09-4935-9686-7a1cb944dd34)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/25bf7883-1957-4b9f-a5e5-1cdf88ad4a08)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/90700224-2aac-40d1-8cf6-ebd2f1eee4dc)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/a67759e6-a522-479c-8a68-c1d870507c7a)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/843dbaf3-e785-4fba-83e2-88a76e2e5169)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/ec644718-8351-4aad-9b00-3455c3c32eac)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/f205b078-ec9f-4990-ad0b-f5e9b5710dcf)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/a15d46bd-efac-4691-94bb-5e709dea142e)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/1f102006-9e93-4c94-a49e-0cd833514272)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/cd44e9db-d5cf-4d6a-a25e-4dae43f17d65)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/9de5221f-9fbf-47c8-a160-f20fe967bc0a)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/52ae2cf0-3643-42fd-b66a-174dff9f1610)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/e8f1696f-91e3-4cdf-889d-7603edd8c976)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/07b3f1df-672a-429f-9514-dba21d007050)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/025990b9-0276-453e-917f-12b69e273990)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/739b43c9-519d-46ae-b394-03355a37698d)




















# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
       
