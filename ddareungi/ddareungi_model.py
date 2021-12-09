#!/usr/bin/env python
# coding: utf-8

# In[1]:


#라이브러리 import
import pandas as pd
import numpy as np
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
import sns as sns
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt
import joblib

# In[2]:


#csv파일 로드 with pandas
train_df = pd.read_csv('./dataset/train.csv')
test_df = pd.read_csv('./dataset/test.csv')

# In[3]:


#기간별 일평균 대기환경 정보 로드
air_quality_2018 = pd.read_csv('../dataset/daily_average_air_environment_2018.csv')
air_quality_2019 = pd.read_csv('../dataset/daily_average_air_environment_2019.csv')
air_quality_2020 = pd.read_csv('../dataset/daily_average_air_environment_2020.csv')

air_quality_2018.describe()

# In[4]:


#2018년 ~ 2020년 merge
air_quality = pd.concat([air_quality_2018, air_quality_2019, air_quality_2020], ignore_index=True)

air_quality = air_quality.drop(['date_time'],axis=1)

air_quality

# In[5]:


#년, 월, 일을 각 feature로 분리하기 위한 작업
train_df['year'] = train_df['date_time'].str[:4].astype(float) - 2017
train_df['month']= train_df['date_time'].str[5:7].astype(float) - 3
train_df['day']= train_df['date_time'].str[8:].astype(float)

train_df

# In[6]:


#기존 데이터 + 미세먼지 정보
train_df = pd.concat([train_df, air_quality],axis=1)

train_df

# In[7]:


#요일 추출
import datetime

#월:1 ~ 일:7
train_df['weekday'] = pd.to_datetime(train_df['date_time']).dt.weekday + 1

train_df.head()

# In[8]:


#일교차 추가
train_df['temp_gap'] = train_df['high_temp'] - train_df['low_temp']

# In[9]:


train_df.describe()



def NMAE(true, pred):
    score = np.mean(np.abs(true-pred) / true)
    return score

#test_df 일자 작업
test_df['year'] = test_df['date_time'].str[:4].astype(float)
test_df['month']= test_df['date_time'].str[5:7].astype(float)
test_df['day']= test_df['date_time'].str[8:].astype(float)

#월:1 ~ 일:7
test_df['weekday'] = pd.to_datetime(test_df['date_time']).dt.weekday + 1

test_df.head()



# <h2>XGboost</h2>



X = train_df.drop(['date_time','number_of_rentals','wind_direction'],axis=1)
Y = train_df['number_of_rentals']



#K-Fold 교차검증
kfold = KFold(n_splits=4, shuffle=True, random_state=97)

for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

# In[14]:


from xgboost import XGBRegressor

# In[15]:


XGB_model = XGBRegressor()

# In[16]:


XGB_model.fit(X_train,
              y_train,
              verbose=True,
              early_stopping_rounds=10,
              eval_metric='rmse',
              eval_set=[(X_test, y_test)]
)

# In[17]:


y_pred = XGB_model.predict(X_test)

# In[18]:


#xgb모델의 NMAE score계산
NMAE(y_test,y_pred)

# In[22]:


score = XGB_model.score(X_test,y_test)
print('정확도: {score:.3f}'.format(score=score))

# In[19]:


#모델 저장
filename='../models/model.pkl'
joblib.dump(XGB_model, filename)

# In[173]:


#user의 input을 가정한 dictionary생성 후 dataframe생성
data = {"sky_condition":3.800, "precipitation_form":0.000, "wind_speed":3.276,"humidity":15.000,"low_temp":12.812, "high_temp":21.000, "Precipitation_Probability":10.000,"year":2021.0,"month":6.0, "day":1.0}

_input = pd.DataFrame(data,index=np.arange(1))

_input

# In[66]:


predictions = [round(value) for value in y_pred]
predictions

# In[157]:


from sklearn.metrics import accuracy_score
accuary = accuracy_score(y_test, predictions)
accuary

# In[24]:


#user의 input의 결과값을 예측했을 때
XGB_model.predict(_input)

# In[159]:


#true, pred비교
for i, v in enumerate(y_pred):
    print(f"{i}번째 true: {y_test[i]}  pred: {v}")

# <h1>상관관계</h1>
# 

# In[67]:


X.corr()

# In[68]:


#heatmap

plt.figure(figsize=(12, 12))
sns.heatmap(X.corr(),annot=True, cmap='YlGnBu')
plt.show()

# In[108]:



