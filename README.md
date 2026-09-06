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
df=pd.read_csv("bmi.csv") 
df.head
```


<img width="351" height="232" alt="image" src="https://github.com/user-attachments/assets/fc11d499-e018-48b6-a666-9a342d754eb3" />


```
df_null_sum=df.isnull().sum() 
df_null_sum
```


<img width="149" height="228" alt="image" src="https://github.com/user-attachments/assets/3905fd7c-a745-4c57-bb51-559205ae3a5f" />


```
df.dropna()
```


<img width="356" height="465" alt="image" src="https://github.com/user-attachments/assets/b575b2f8-77cf-49e5-9045-17bb44ff8457" />

```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0) 
max_vals
```


<img width="141" height="163" alt="image" src="https://github.com/user-attachments/assets/be80a839-e409-480c-b442-6b26e51d848b" />



```
from sklearn.preprocessing import StandardScaler 
df1=pd.read_csv("bmi.csv") 
df1.head()
```



<img width="321" height="231" alt="image" src="https://github.com/user-attachments/assets/b4dae091-e0a9-4394-9ae0-6c26868ee5cd" />



```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']]) 
df1.head(10)
```


<img width="378" height="400" alt="image" src="https://github.com/user-attachments/assets/b63406b6-5f72-4b81-9423-4c1fff3ebc81" />


```
#MIN-MAX SCALING: 
from sklearn.preprocessing import MinMaxScaler 
scaler=MinMaxScaler() 
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df.head(10)

```


<img width="397" height="412" alt="image" src="https://github.com/user-attachments/assets/5cd5bc10-5c64-4227-a33d-37e8a7ff9299" />



```
from sklearn.preprocessing import MaxAbsScaler 
scaler = MaxAbsScaler() 
df3=pd.read_csv("bmi.csv") 
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']]) 
df
```



<img width="333" height="239" alt="image" src="https://github.com/user-attachments/assets/a2faba71-7cd1-4357-95f6-5e46dec85e47" />

```
from sklearn.preprocessing import RobustScaler 
scaler = RobustScaler() 
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']]) 
df3.head()
```


<img width="367" height="225" alt="image" src="https://github.com/user-attachments/assets/77a088a9-5627-4b34-ba5a-91483878fde3" />



```
df=pd.read_csv("income(1) (1).csv") 
df.info()
```


<img width="405" height="406" alt="image" src="https://github.com/user-attachments/assets/19c07662-f22d-4e2f-af14-05360ac48396" />




```
df_null_sum=df.isnull().sum() 
df_null_sum
```


<img width="178" height="551" alt="image" src="https://github.com/user-attachments/assets/ce72f311-2024-43fe-871f-e1e82da344e1" />



```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```



<img width="946" height="469" alt="image" src="https://github.com/user-attachments/assets/4e001951-eeb1-488e-a7e9-75b6e283620f" />




```
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories. 
df[categorical_columns]
```


<img width="834" height="472" alt="image" src="https://github.com/user-attachments/assets/b0df5571-7f8c-479d-9670-90f102a13440" />


```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)

```


<img width="394" height="94" alt="image" src="https://github.com/user-attachments/assets/0ba928da-f1eb-44e1-b601-a037234bbdff" />



```
y_pred = rf.predict(X_test)
df=pd.read_csv("income(1) (1).csv") 
df.info()
```


<img width="424" height="405" alt="image" src="https://github.com/user-attachments/assets/9541dd98-fd40-4b45-ae89-d6b78ad6444c" />



```
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry'] 
df[categorical_columns] = df[categorical_columns].astype('category') 
df[categorical_columns]
```


<img width="966" height="474" alt="image" src="https://github.com/user-attachments/assets/aa89b550-d9f6-4017-b928-b864d634d043" />



```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes) 
df[categorical_columns]
```


<img width="890" height="465" alt="image" src="https://github.com/user-attachments/assets/b4d0fe0f-ada8-4150-a066-979bbc68d0df" />


```
X = df.drop(columns=['SalStat']) 
y = df['SalStat'] 
k_chi2 = 6 
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2) 
X_chi2 = selector_chi2.fit_transform(X, y) 
selected_features_chi2 = X.columns[selector_chi2.get_support()] 
print("Selected features using chi-square test:") 
print(selected_features_chi2)

```


<img width="688" height="92" alt="image" src="https://github.com/user-attachments/assets/a55b7d2c-2241-4098-8a9d-f4302b1897b4" />


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif 
from sklearn.model_selection import train_test_split 
# Importing the missing function 
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss', 'hoursperweek'] 
X = df[selected_features]
y = df['SalStat'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 
rf = RandomForestClassifier(n_estimators=100, random_state=42) 
rf.fit(X_train, y_train)
```


<img width="421" height="88" alt="image" src="https://github.com/user-attachments/assets/15c622d8-2ae5-468d-aa6a-0c0f540eb746" />

```
y_pred = rf.predict(X_test) 
from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(y_test, y_pred) 
print(f"Model accuracy using selected features: {accuracy}")
```

<img width="579" height="35" alt="image" src="https://github.com/user-attachments/assets/c54fd5f0-272b-4f66-ba79-9012af27ff21" />


```
import numpy as np 
import pandas as pd
from skfeature.function.similarity_based import fisher_score 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ] 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes) 
# @title 
df[categorical_columns]
```


<img width="833" height="475" alt="image" src="https://github.com/user-attachments/assets/89720fbc-b186-4415-9c16-557dc40a1126" />



```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
k_anova = 5 
selector_anova = SelectKBest(score_func=f_classif,k=k_anova) 
X_anova = selector_anova.fit_transform(X, y)
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:") 
print(selected_features_anova)
```


<img width="767" height="76" alt="image" src="https://github.com/user-attachments/assets/29b3d77b-a044-475e-b4a3-387a6b74733b" />




```
import pandas as pd 
from sklearn.feature_selection import RFE 
from sklearn.linear_model import LogisticRegression 
df=pd.read_csv("income(1) (1).csv") # List of categorical columns 
categorical_columns = [ 'JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry' ] # Convert the categorical columns to category dtype 
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```



<img width="839" height="466" alt="image" src="https://github.com/user-attachments/assets/2ad81fd4-b80f-45b6-ac60-68c0b63b9c2a" />



```
X = df.drop(columns=['SalStat']) 
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select) 
rfe.fit(X, y)
```



<img width="339" height="196" alt="image" src="https://github.com/user-attachments/assets/ca799be7-776f-40b0-ad9f-e07f9eb6284e" />




# RESULT:
  To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file was done and executed sucessesfully
