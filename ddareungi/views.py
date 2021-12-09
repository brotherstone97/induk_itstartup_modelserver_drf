from django.shortcuts import render
from rest_framework.views import APIView
from .load_model import predict_rental


# Create your views here.
class Weather(APIView):
    def post(self, request):
        # request.data == 단기예보의 json데이터
        raw_weather_data = request.data
        predict_rental(raw_weather_data)
