from __future__ import annotations

from django.contrib.auth.models import AbstractUser, UserManager
from django.db import models


class AccountUserManager(UserManager):
    """自定义用户管理器，确保超级管理员默认拥有 admin 角色。"""

    def create_superuser(self, username, email=None, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)
        extra_fields.setdefault("is_active", True)
        extra_fields.setdefault("role", User.Role.ADMIN)
        return super().create_superuser(username=username, email=email, password=password, **extra_fields)


class User(AbstractUser):
    """系统用户模型，仅保留毕业设计所需的管理员与普通用户两类角色。"""

    class Role(models.TextChoices):
        ADMIN = "admin", "管理员"
        USER = "user", "普通用户"

    role = models.CharField(max_length=20, choices=Role.choices, default=Role.USER, verbose_name="角色")
    real_name = models.CharField(max_length=50, blank=True, default="", verbose_name="姓名")
    phone = models.CharField(max_length=20, blank=True, default="", verbose_name="手机号")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    objects = AccountUserManager()

    class Meta:
        db_table = "sys_user"
        verbose_name = "用户"
        verbose_name_plural = "用户"

    def __str__(self) -> str:
        return f"{self.username}({self.role})"
