import datetime

import joblib
import pandas as pd
import numpy as np
from pandas import to_datetime

# loaded_model = joblib.load('../models/model.pkl')

# user의 input을 가정한 더미 데이터
# data = {"sky_condition": 3.800,
#         "precipitation_form": 0.000, "wind_speed": 3.276,
#         "humidity": 15.000, "low_temp": 12.812, "high_temp": 21.000,
#         "Precipitation_Probability": 10.000, "year": 2021.0,
#         "month": 6.0, "day": 1.0, "PM10": 71.45, "PM2.5": 21.04,
#         "weekday": 2}
# data['temp_gap'] = data['high_temp'] - data['low_temp']
# _input = pd.DataFrame(data, index=np.arange(1))

# SKY(하늘상태), POP(강수확률), TMN(일 최저기온), PTY(강수형태), REH(습도), WSD(풍속), TMX(일 최고기온)에 대응하는 column_name
column_name_list = ["sky_condition", "Precipitation_Probability", "low_temp", "precipitation_form", "humidity",
                    "wind_speed", "high_temp"]

final_weather = {}

now = datetime.datetime.now()

# spring으로 부터 날씨 정보와 미세먼지 정보를 받음
def predict_rental(weather):
    #기존 키값을 model의 column으로 변경하고 value값을 float로 변환하여 저장
    for i, key in enumerate(weather):
        final_weather[column_name_list[i]] = float(weather[key])


    # 월:1 ~ 일:7
    final_weather['year'], final_weather['month'], final_weather['day'] = now.year, now.month, now.day
    final_weather['weekday'] = datetime.datetime.today().weekday() + 1


    return final_weather
    # input값에 대한 최종 대여량 예측값
    # result = loaded_model.predict(weather)
