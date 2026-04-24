from __future__ import annotations

import threading
from pathlib import Path

from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils import timezone

from fitness_action_eval.model_options import resolve_pose_model_path
from fitness_action_eval.pipeline import save_pose_template

from .models import ActionCategory, FileAsset, TemplateVideo

_TEMPLATE_BUILD_REGISTRY: dict[int, dict[str, object]] = {}
_TEMPLATE_BUILD_LOCK = threading.Lock()


def ensure_default_baduanjin_category() -> ActionCategory:
    """确保系统内始终存在默认的八段锦动作分类。"""

    category, created = ActionCategory.objects.get_or_create(
        code="BADUANJIN",
        defaults={
            "name": "八段锦",
            "description": "八段锦动作模板分类",
            "is_active": True,
        },
    )
    changed = []
    if category.name != "八段锦":
        category.name = "八段锦"
        changed.append("name")
    if category.description != "八段锦动作模板分类":
        category.description = "八段锦动作模板分类"
        changed.append("description")
    if not category.is_active:
        category.is_active = True
        changed.append("is_active")
    if changed and not created:
        changed.append("updated_at")
        category.save(update_fields=changed)
    return category


def register_source_asset(template: TemplateVideo) -> FileAsset:
    """登记模板源视频。"""

    file_name = Path(template.source_video.name).name
    return FileAsset.objects.create(
        biz_type=FileAsset.BizType.TEMPLATE_SOURCE,
        biz_id=template.id,
        file_name=file_name,
        file_path=template.source_video.path,
        file_type=Path(file_name).suffix.lower().lstrip("."),
        file_size=template.source_video.size or 0,
    )


def _update_template_progress(template_id: int, percent: int, text: str) -> None:
    """模板提取过程中回写进度条。"""

    TemplateVideo.objects.filter(id=template_id).update(
        progress_percent=max(0, min(100, int(percent))),
        progress_text=str(text),
        updated_at=timezone.now(),
    )


def build_template_file(template: TemplateVideo, progress_callback=None) -> TemplateVideo:
    """调用算法端从标准视频中提取姿态模板文件。"""

    generated_dir = Path(settings.MEDIA_ROOT) / "template_center" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = generated_dir / f"template_{template.id}_{template.version}.npz"

    try:
        save_pose_template(
            ref_video=template.source_video.path,
            task_model=resolve_pose_model_path(template.pose_model),
            num_poses=1,
            smooth_window=template.smooth_window,
            template_path=str(output_path),
            frame_stride=template.frame_stride,
            preview=False,
            progress_callback=progress_callback,
        )
        template.template_file_path = str(output_path)
        template.status = TemplateVideo.Status.READY
        template.progress_percent = 100
        template.progress_text = "模板生成完成"
        template.build_error = ""
        template.save(update_fields=["template_file_path", "status", "progress_percent", "progress_text", "build_error", "updated_at"])

        FileAsset.objects.update_or_create(
            biz_type=FileAsset.BizType.TEMPLATE_FILE,
            biz_id=template.id,
            defaults={
                "file_name": output_path.name,
                "file_path": str(output_path),
                "file_type": "npz",
                "file_size": output_path.stat().st_size if output_path.exists() else 0,
            },
        )
    except Exception as exc:
        template.status = TemplateVideo.Status.FAILED
        template.progress_percent = 100
        template.progress_text = "模板生成失败"
        template.build_error = str(exc)
        template.save(update_fields=["status", "progress_percent", "progress_text", "build_error", "updated_at"])
        raise
    return template


def _run_template_build_worker(template_id: int) -> None:
    """后台线程：生成模板文件。"""

    close_old_connections()
    try:
        template = TemplateVideo.objects.get(id=template_id)
        build_template_file(
            template,
            progress_callback=lambda percent, text: _update_template_progress(template_id, percent, text),
        )
    except TemplateVideo.DoesNotExist:
        pass
    except Exception as exc:
        TemplateVideo.objects.filter(id=template_id).update(
            status=TemplateVideo.Status.FAILED,
            progress_percent=100,
            progress_text="模板生成失败",
            build_error=str(exc),
            updated_at=timezone.now(),
        )
    finally:
        with _TEMPLATE_BUILD_LOCK:
            _TEMPLATE_BUILD_REGISTRY.pop(template_id, None)
        close_old_connections()


def start_template_build(template: TemplateVideo) -> TemplateVideo:
    """启动模板生成线程。"""

    TemplateVideo.objects.filter(id=template.id).update(
        status=TemplateVideo.Status.BUILDING,
        progress_percent=0,
        progress_text="等待生成模板",
        build_error="",
        updated_at=timezone.now(),
    )
    thread = threading.Thread(
        target=_run_template_build_worker,
        args=(template.id,),
        name=f"template-build-{template.id}",
        daemon=True,
    )
    with _TEMPLATE_BUILD_LOCK:
        _TEMPLATE_BUILD_REGISTRY[template.id] = {"thread": thread}
    thread.start()
    template.refresh_from_db()
    return template


def delete_template_bundle(template: TemplateVideo) -> None:
    """删除模板记录及其关联文件。"""

    source_storage = template.source_video.storage
    source_name = template.source_video.name
    asset_paths = {
        item.file_path
        for item in FileAsset.objects.filter(
            biz_id=template.id,
            biz_type__in=[FileAsset.BizType.TEMPLATE_SOURCE, FileAsset.BizType.TEMPLATE_FILE],
        )
        if item.file_path
    }
    if template.template_file_path:
        asset_paths.add(template.template_file_path)

    with transaction.atomic():
        FileAsset.objects.filter(
            biz_id=template.id,
            biz_type__in=[FileAsset.BizType.TEMPLATE_SOURCE, FileAsset.BizType.TEMPLATE_FILE],
        ).delete()
        template.delete()

    for file_path in asset_paths:
        path = Path(file_path)
        if path.exists():
            path.unlink()

    if source_name:
        try:
            source_storage.delete(source_name)
        except Exception:
            pass
