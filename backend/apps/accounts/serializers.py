from __future__ import annotations

from django.contrib.auth import authenticate
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken

from .models import User


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = (
            "id",
            "username",
            "real_name",
            "phone",
            "email",
            "role",
            "is_active",
            "last_login",
            "date_joined",
        )


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True, style={"input_type": "password"})

    def validate(self, attrs):
        request = self.context.get("request")
        username = attrs.get("username")
        password = attrs.get("password")
        user = authenticate(request=request, username=username, password=password)
        if user is None:
            raise serializers.ValidationError("用户名或密码错误。")
        if not user.is_active:
            raise serializers.ValidationError("当前账号已被禁用。")
        attrs["user"] = user
        return attrs

    @staticmethod
    def build_token_payload(user: User) -> dict:
        refresh = RefreshToken.for_user(user)
        return {
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "user": UserProfileSerializer(user).data,
        }


class RefreshTokenSerializer(serializers.Serializer):
    refresh = serializers.CharField()

    def validate(self, attrs):
        try:
            refresh = RefreshToken(attrs["refresh"])
        except Exception as exc:
            raise serializers.ValidationError("刷新令牌无效。") from exc
        attrs["access"] = str(refresh.access_token)
        return attrs
