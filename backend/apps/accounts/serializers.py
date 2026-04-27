from __future__ import annotations

from django.contrib.auth import authenticate
from rest_framework import serializers
from rest_framework_simplejwt.tokens import RefreshToken

from .models import User


class UserProfileSerializer(serializers.ModelSerializer):
    """登录后返回给前端的当前用户资料。"""

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
    """用户名密码登录。"""

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
        """返回前端需要缓存的 JWT 与用户信息。"""

        refresh = RefreshToken.for_user(user)
        return {
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "user": UserProfileSerializer(user).data,
        }


class RefreshTokenSerializer(serializers.Serializer):
    """使用 refresh token 续签 access token。"""

    refresh = serializers.CharField()

    def validate(self, attrs):
        try:
            refresh = RefreshToken(attrs["refresh"])
        except Exception as exc:
            raise serializers.ValidationError("刷新令牌无效。") from exc
        attrs["access"] = str(refresh.access_token)
        return attrs


class AdminUserListSerializer(serializers.ModelSerializer):
    """管理员端用户列表序列化器。"""

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
            "is_superuser",
            "last_login",
            "date_joined",
        )


class AdminUserCreateSerializer(serializers.ModelSerializer):
    """管理员创建用户时的参数校验。"""

    password = serializers.CharField(write_only=True, min_length=6)

    class Meta:
        model = User
        fields = ("username", "password", "real_name", "phone", "email", "role", "is_active")

    def validate_username(self, value: str) -> str:
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("用户名已存在。")
        return value

    def create(self, validated_data):
        password = validated_data.pop("password")
        role = validated_data.get("role", User.Role.USER)
        user = User(**validated_data)
        user.set_password(password)
        user.is_staff = role == User.Role.ADMIN
        user.is_superuser = role == User.Role.ADMIN
        user.save()
        return user


class AdminUserUpdateSerializer(serializers.ModelSerializer):
    """管理员编辑用户基础资料、角色和启用状态。"""

    class Meta:
        model = User
        fields = ("real_name", "phone", "email", "role", "is_active")

    def validate(self, attrs):
        request = self.context.get("request")
        instance: User = self.instance
        if request is not None and instance and request.user.id == instance.id:
            if "role" in attrs and attrs["role"] != instance.role:
                raise serializers.ValidationError({"role": ["不能在此处修改当前管理员自己的角色。"]})
            if "is_active" in attrs and attrs["is_active"] != instance.is_active:
                raise serializers.ValidationError({"is_active": ["不能在此处禁用当前管理员自己的账号。"]})
        return attrs

    def update(self, instance, validated_data):
        role = validated_data.get("role", instance.role)
        for field, value in validated_data.items():
            setattr(instance, field, value)
        instance.is_staff = role == User.Role.ADMIN
        instance.is_superuser = role == User.Role.ADMIN
        instance.save()
        return instance


class AdminUserResetPasswordSerializer(serializers.Serializer):
    """管理员重置用户密码。"""

    new_password = serializers.CharField(min_length=6, max_length=128)
