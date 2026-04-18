from django.urls import path

from .views import (
    category_list_view,
    template_build_view,
    template_delete_view,
    template_detail_view,
    template_list_view,
    template_upload_view,
)


urlpatterns = [
    path("categories/", category_list_view, name="template-category-list"),
    path("templates/", template_list_view, name="template-list"),
    path("templates/upload/", template_upload_view, name="template-upload"),
    path("templates/<int:template_id>/", template_detail_view, name="template-detail"),
    path("templates/<int:template_id>/build/", template_build_view, name="template-build"),
    path("templates/<int:template_id>/delete/", template_delete_view, name="template-delete"),
]
