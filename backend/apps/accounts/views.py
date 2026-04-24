from __future__ import annotations

from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

from config.api_response import api_error, api_success

from .serializers import LoginSerializer, RefreshTokenSerializer, UserProfileSerializer


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
