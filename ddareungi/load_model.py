import joblib
import pandas as pd
import numpy as np

loaded_model = joblib.load('../models/model.pkl')


#user의 input을 가정한 더미 데이터
data = {"sky_condition":3.800, 
        "precipitation_form":0.000, "wind_speed":3.276,
        "humidity":15.000,"low_temp":12.812, "high_temp":21.000, 
        "Precipitation_Probability":10.000,"year":2021.0,
        "month":6.0, "day":1.0, "PM10":71.45,"PM2.5":21.04,
       "weekday":2}
data['temp_gap'] = data['high_temp'] - data['low_temp']
_input = pd.DataFrame(data,index=np.arange(1))




#spring으로 부터 날씨 정보와 미세먼지 정보를 받음
# def predict_rental(weather):
    




#input값에 대한 최종 대여량 예측값
loaded_model.predict(_input)
