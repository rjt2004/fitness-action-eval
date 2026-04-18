from django.urls import path

from .views import live_session_detail_view, live_session_list_view, live_session_start_view, live_session_stop_view


urlpatterns = [
    path("sessions/", live_session_list_view, name="live-session-list"),
    path("sessions/start/", live_session_start_view, name="live-session-start"),
    path("sessions/<int:session_id>/", live_session_detail_view, name="live-session-detail"),
    path("sessions/<int:session_id>/stop/", live_session_stop_view, name="live-session-stop"),
]
