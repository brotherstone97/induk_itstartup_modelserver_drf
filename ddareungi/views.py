from rest_framework.views import APIView
from rest_framework.response import Response
from .load_model import predict_rental


# Create your views here.
class Weather(APIView):
    def post(self, request):
        # request.data == 단기예보의 json데이터
        # request.data예시: {"sky_condition":3.800,
        #         "precipitation_form":0.000, "wind_speed":3.276,
        #         "humidity":15.000,"low_temp":12.812, "high_temp":21.000,
        #         "Precipitation_Probability":10.000,"year":2021.0,
        #         "month":6.0, "day":1.0, "PM10":71.45,"PM2.5":21.04,
        #        "weekday":2}

        raw_weather_data = request.data
        result = predict_rental(raw_weather_data)

        return Response(result)

