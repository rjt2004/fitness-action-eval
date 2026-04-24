from rest_framework.permissions import BasePermission


class IsAdminRole(BasePermission):
    """仅允许管理员访问的权限类。"""

    message = "只有管理员可以执行该操作。"

    def has_permission(self, request, view) -> bool:
        user = request.user
        return bool(user and user.is_authenticated and getattr(user, "role", "") == "admin")
