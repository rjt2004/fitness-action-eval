from django.urls import path

from .views import health_check, pose_model_options


urlpatterns = [
    path("health/", health_check, name="health-check"),
    path("pose-model-options/", pose_model_options, name="pose-model-options"),
]
