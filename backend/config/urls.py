from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/auth/", include("apps.accounts.urls")),
    path("api/system/", include("apps.system.urls")),
    path("api/template-center/", include("apps.template_manager.urls")),
    path("api/evaluation-center/", include("apps.evaluation_center.urls")),
    path("api/live-session/", include("apps.live_session.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
