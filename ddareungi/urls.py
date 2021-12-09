from django.urls import path
from . import views


urlpatterns=[
    path('', views.Weather.post.as_view())
]