from django.urls import path

from .views import current_user_view, login_view, refresh_token_view


urlpatterns = [
    path("login/", login_view, name="auth-login"),
    path("refresh/", refresh_token_view, name="auth-refresh"),
    path("me/", current_user_view, name="auth-me"),
]
