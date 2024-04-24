# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
       import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/19ce53e4-aa6f-4a8e-b485-605cbc158803)

```
df.dropna()
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/ebcb961f-f932-4bd8-b0f0-f4c67c780930)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/a1adf4a4-b158-4282-904b-6b288e2b08a1)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/80ea03e4-16dc-4251-b7bb-7adea8701afb)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/39a888ac-74f2-4579-8150-bc0676073f92)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/944bf054-ede4-469b-a35a-44ebc30fd1e1)

```
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/f5b2b200-2eea-45f8-8da7-a2163b42a3f4)

```
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/ab602c01-b9f8-4d58-92ba-b15fbe57c736)

```

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/b11707a2-bbc0-48a1-9ad6-c79146d4e927)

```
data.isnull().sum()
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/b51136c8-f449-4c3d-a771-7b4b50d28992)

```
missing=data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/1e9a4a91-90e0-4c85-af42-dec1bcfd7dc7)

```
data2 = data.dropna(axis=0)
data2
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/f7f4f6fc-f12a-429c-bc3b-550ecfda9fa4)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/891d81ee-5813-4166-91c0-9e76e1ef589a)

```

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/48099ae9-c2ff-4d42-9749-1f240dd9be7a)

```
data2
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/bca18dac-e89e-4daf-a2f0-648c10b5cbfc)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/b87309ea-0a4a-4c6c-909f-3fa29f31e719)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/1d3f758a-f362-403a-8be4-cf9feb648c00)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/14b98d3a-d7b3-48a6-8330-44c83e847080)

```
y=new_data['SalStat'].values
print(y)
```
[0 0 1 ... 0 0 0]

```
x = new_data[features].values
print(x)
```

![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/8567c08d-b7c7-4add-995b-f794ace35711)

```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/fb751c72-a640-4049-b63e-57e2559e268c)

```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/adcbeaba-4d01-427e-823a-a0c6523dea5d)

```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
0.8392087523483258

```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
Misclassified samples: 1455

```
data.shape

(31978, 13)
```

## FEATURE SELECTION TECHNIQUES

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/1cc87cbb-8275-41c9-9dc7-0637790d2bbb)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/0f2cafaf-7b43-489c-a7d7-79c30ce485df)

```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/bd17a761-3598-4fdd-8372-a037221beaae)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/Nishanth-018/EXNO-4-DS/assets/149347651/e2c83404-4293-436f-9179-6e009886364d)


## RESULT:
       To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successful.
