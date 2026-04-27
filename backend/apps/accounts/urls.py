from django.urls import path

from .views import (
    admin_user_detail_view,
    admin_user_list_create_view,
    admin_user_reset_password_view,
    current_user_view,
    login_view,
    refresh_token_view,
)


urlpatterns = [
    path("login/", login_view, name="auth-login"),
    path("refresh/", refresh_token_view, name="auth-refresh"),
    path("me/", current_user_view, name="auth-me"),
    path("users/", admin_user_list_create_view, name="admin-user-list-create"),
    path("users/<int:user_id>/", admin_user_detail_view, name="admin-user-detail"),
    path("users/<int:user_id>/reset-password/", admin_user_reset_password_view, name="admin-user-reset-password"),
]
