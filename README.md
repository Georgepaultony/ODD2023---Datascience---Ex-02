# Ex02-Outlier
You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

(i) Using IQR detect weight outliers and print them

(ii) Using IQR, detect height outliers and print them


## AIM:

TO detect and remove the outliers in the given data set and save the final data.

## EXPLANATION:

An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.

## ALGORITHM:

STEP 1 Read the given Data

STEP 2 Get the information about the data

STEP 3 Detect the Outliers using IQR method and Z score

STEP 4 Remove the outliers

## CODE AND OUTPUT

```
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np
```

![266832635-52863607-9403-4d68-b6d1-b4ecc898d0ee](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/34b402a1-13e3-4f83-9e19-590e56be2e31)

```
from google.colab import files
uploaded = files.upload()
```

![266832644-f13a32b6-98da-4efa-8376-309f0f8a404f](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/eb7a4115-4708-4b03-92a4-cfa25a447e1f)

```
df = pd.read_csv("bhp.csv")
q1 = df['price_per_sqft'].quantile(0.25)
q2 = df['price_per_sqft'].quantile(0.5)
q3 = df['price_per_sqft'].quantile(0.75)
iqr = q3-q1
iqr
```

![266832660-7a2273b0-b82d-471b-81b6-4180c23ba2a6](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/b2df9dff-d7f8-4770-ad56-fb376999dfca)

```
low = q1-1.5*iqr
low
```

![266832671-5f0fd443-b383-42c1-8317-042ef101b6f8](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/6d19bf15-2d5b-40ec-a291-bd820e55595a)

```
high = q3+1.5*iqr
high
```

![266832682-31c43390-13ad-4bae-8702-c8a6410c833b](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/e3202fba-74b0-4ec7-ba69-03af8f92db64)

```
df = df[((df['price_per_sqft']>=low) & (df['price_per_sqft']<=high))]
df
```

![266832706-3989396c-953c-4aa3-b6b3-e86bee7531a6](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/5794b895-0bed-437c-80fb-60b165e9ac00)

```
z = np.abs(stats.zscore(df['price_per_sqft']))
z
```

![266832720-1aefd877-e4b5-4301-831d-c7d69b4e9ece](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/faf73dd2-f3e3-4164-bc7f-193401cf8f11)

```
df1 = df[z<3]
df1
```

![266832727-623f4d88-f4ca-42a3-8703-eb53c782a1b8](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/332c0a59-53ff-43bf-9149-e4927c70dbae)

```
from google.colab import files
uploaded = files.upload()
```

![266832775-e0135abc-3cf7-49c2-ae1e-68e4dccda3f2](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/c2f2f32b-4787-4abe-9f4e-246295808947)

```
df = pd.read_csv("height_weight.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```

![266832796-1b06ad91-b99c-40af-aacf-6840c589e940](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/fe2dba3b-0325-42a1-a323-6ec47e22a7da)

```
low = q1 - 1.5*iqr
low
```

![266832813-cb38d668-c592-407a-8211-e8efa20f1c71](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/e2faadfa-7bbd-45d5-8657-ed65d7d41892)

```
high = q3+1.5*iqr
high
```

![266832825-a461b70d-2de3-45fe-b0a4-0f89cf4d9d65](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/4acb6ab2-ad07-4108-a8b7-59fff4f39f61)

```
df = df[((df['height'] >=low) & (df['height']<= high))]
df
```

![266832841-a25f4a99-4b7e-4464-903a-f88159b2feb8](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/60570a0d-e803-477b-8833-928e31b996f8)

```
z = np.abs(stats.zscore(df['height']))
z
```

![266832874-7226af9a-1735-4ef8-b37d-b7b31980c739](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/9ac2c517-5a08-4e58-a832-3cf7396ceb00)


```
df1 = df[z<3]
df1
```

![266832892-cc06c6ff-e4bb-4c1f-a7ce-9a57009d069c](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/74e80507-9888-4af6-a825-b4fdd857d34e)

```
df = pd.read_csv("height_weight.csv")
q1 = df['weight'].quantile(0.25)
q2 = df['weight'].quantile(0.5)
q3 = df['weight'].quantile(0.75)
iqr = q3-q1
iqr
```

![266832907-76aad33e-48ff-4cd7-a397-ad8ad0058cf5](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/b7ea150d-8049-478a-b5c8-3a98cbe94c3c)

```
low = q1 - 1.5*iqr
low
```

![266832962-968e03c9-dd25-4d7c-b0aa-a501e623ba80](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/ec3317a8-66ef-47e1-ac6d-fc79f55b1834)

```
high = q3 + 1.5*iqr
high
```

![266832993-36b2c610-b542-4237-abf3-7ca393bb21fe](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/8e2a4938-f12d-4f62-aa03-404914dbb03b)

```
df1 = df[((df['weight'] >=low) & (df['weight']<= high))]
df1
```

![266833027-971353a7-6646-474e-aac7-c1b5a77282e0](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/2d5db02e-6e6e-4c29-b81e-db4f25b9003b)


```
z = np.abs(stats.zscore(df1['weight']))
z
```

![266833095-4fdc96bd-cd1b-4bce-a2d8-446c09895bae](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/4d88c4e5-acbc-4ecf-a6a9-4cf3be44b68e)


```
df2 = df1[z<3]
df2
```

![266833108-3ca5ddd2-6b99-4b31-a856-4796d5d39e03](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/7b6d44ab-edfc-479f-892c-d3192ea3c887)


```
from google.colab import files
uploaded = files.upload()
```

![266833127-e2a30f94-50bb-4484-a73c-94800feb6df6](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/85ed05ad-c5e6-4f02-91a9-fd147eef62f6)


```
df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
iqr = q3-q1
iqr
```

![266833149-22710613-4eb6-46be-8e3a-98325e34938a](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/609bcecd-32e6-4cb9-a946-9459788fc50c)

```
low = q1 - 1.5*iqr
low
```

![266833169-12560539-2134-4e2a-a25f-9288935a8b21](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/449061d2-8e75-479d-8756-dd0eafae1366)


```
high = q3 + 1.5*iqr
high
```

![266833192-347d027a-9c53-4488-aaab-68744e98bfce](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/9e91051c-ae33-4f9e-aafc-a74229fc7c60)


```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```

![266833224-5edaf9be-6e7b-436f-a3eb-d4a6782e988b](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/69dd6ed6-9d16-41de-a9d2-b19b939cc9b1)


```
\z = np.abs(stats.zscore(df['height']))
z
```

![266833257-813dc0d0-9778-4e29-83d3-61f70acdc041](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/615c37f9-97e3-4cbc-8c1f-bb2958a08c4f)


```
df1 = df[z<3]
df1
```

![266833280-de4770d5-9637-40da-a34a-1fff5b4663ac](https://github.com/Georgepaultony/ODD2023---Datascience---Ex-02/assets/120088748/71dd4701-7d72-4873-8bb7-11d023bb0c28)


## RESULT:

The given datasets are read and outliers are detected and are removed using IQR and z-score methods.







