from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User


@admin.register(User)
class AccountUserAdmin(UserAdmin):
    list_display = ("id", "username", "real_name", "role", "is_active", "is_staff", "last_login")
    list_filter = ("role", "is_active", "is_staff", "is_superuser")
    search_fields = ("username", "real_name", "phone")
    ordering = ("id",)

    fieldsets = UserAdmin.fieldsets + (
        ("业务信息", {"fields": ("role", "real_name", "phone")}),
        ("时间信息", {"fields": ("created_at", "updated_at")}),
    )
    readonly_fields = ("created_at", "updated_at", "last_login", "date_joined")

    add_fieldsets = UserAdmin.add_fieldsets + (
        ("业务信息", {"fields": ("role", "real_name", "phone")}),
    )
