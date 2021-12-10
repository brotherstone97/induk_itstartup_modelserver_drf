from django.urls import path
from . import views

urlpatterns = [
    path('predict', views.Weather.as_view())
]
