from django.urls import path

from .views import (
    evaluation_task_create_view,
    evaluation_task_detail_view,
    evaluation_task_hints_view,
    evaluation_task_list_view,
    evaluation_task_phases_view,
)


urlpatterns = [
    path("tasks/", evaluation_task_list_view, name="evaluation-task-list"),
    path("tasks/create/", evaluation_task_create_view, name="evaluation-task-create"),
    path("tasks/<int:task_id>/", evaluation_task_detail_view, name="evaluation-task-detail"),
    path("tasks/<int:task_id>/phases/", evaluation_task_phases_view, name="evaluation-task-phases"),
    path("tasks/<int:task_id>/hints/", evaluation_task_hints_view, name="evaluation-task-hints"),
]
