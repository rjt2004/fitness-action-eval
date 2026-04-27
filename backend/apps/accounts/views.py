from __future__ import annotations

from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

from config.api_response import api_error, api_success

from .models import User
from .permissions import IsAdminRole
from .serializers import (
    AdminUserCreateSerializer,
    AdminUserListSerializer,
    AdminUserResetPasswordSerializer,
    AdminUserUpdateSerializer,
    LoginSerializer,
    RefreshTokenSerializer,
    UserProfileSerializer,
)


@api_view(["POST"])
@authentication_classes([])
@permission_classes([AllowAny])
def login_view(request):
    """账号密码登录接口。"""

    serializer = LoginSerializer(data=request.data, context={"request": request})
    if not serializer.is_valid():
        return api_error(message="登录失败", data=serializer.errors, status_code=400)
    payload = serializer.build_token_payload(serializer.validated_data["user"])
    return api_success(data=payload, message="登录成功")


@api_view(["POST"])
@authentication_classes([])
@permission_classes([AllowAny])
def refresh_token_view(request):
    """刷新 access token。"""

    serializer = RefreshTokenSerializer(data=request.data)
    if not serializer.is_valid():
        return api_error(message="刷新失败", data=serializer.errors, status_code=400)
    return api_success(data={"access": serializer.validated_data["access"]}, message="刷新成功")


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def current_user_view(request):
    """返回当前登录用户资料。"""

    return api_success(data=UserProfileSerializer(request.user).data, message="获取当前用户成功")


@api_view(["GET", "POST"])
@permission_classes([IsAuthenticated, IsAdminRole])
def admin_user_list_create_view(request):
    """管理员查看用户列表或创建新用户。"""

    if request.method == "GET":
        queryset = User.objects.all().order_by("-id")
        payload = AdminUserListSerializer(queryset, many=True).data
        return api_success(data=payload, message="获取用户列表成功")

    serializer = AdminUserCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return api_error(message="创建用户失败", data=serializer.errors, status_code=400)
    user = serializer.save()
    return api_success(data=AdminUserListSerializer(user).data, message="创建用户成功", status_code=201)


@api_view(["PATCH", "DELETE"])
@permission_classes([IsAuthenticated, IsAdminRole])
def admin_user_detail_view(request, user_id: int):
    """管理员编辑或删除用户。"""

    target = get_object_or_404(User, id=user_id)

    if request.method == "PATCH":
        serializer = AdminUserUpdateSerializer(target, data=request.data, partial=True, context={"request": request})
        if not serializer.is_valid():
            return api_error(message="更新用户失败", data=serializer.errors, status_code=400)
        user = serializer.save()
        return api_success(data=AdminUserListSerializer(user).data, message="更新用户成功")

    if request.user.id == target.id:
        return api_error(message="不能删除当前登录的管理员账号。", status_code=400)

    target.delete()
    return api_success(data={"user_id": user_id}, message="删除用户成功")


@api_view(["POST"])
@permission_classes([IsAuthenticated, IsAdminRole])
def admin_user_reset_password_view(request, user_id: int):
    """管理员重置指定用户的密码。"""

    target = get_object_or_404(User, id=user_id)
    serializer = AdminUserResetPasswordSerializer(data=request.data)
    if not serializer.is_valid():
        return api_error(message="重置密码失败", data=serializer.errors, status_code=400)

    target.set_password(serializer.validated_data["new_password"])
    target.save(update_fields=["password"])
    return api_success(data={"user_id": user_id}, message="重置密码成功")
