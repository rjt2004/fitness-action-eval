from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.db import transaction

from fitness_action_eval.pipeline import save_pose_template

from .models import ActionCategory, FileAsset, TemplateVideo


def ensure_default_baduanjin_category() -> ActionCategory:
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
    file_name = Path(template.source_video.name).name
    return FileAsset.objects.create(
        biz_type=FileAsset.BizType.TEMPLATE_SOURCE,
        biz_id=template.id,
        file_name=file_name,
        file_path=template.source_video.path,
        file_type=Path(file_name).suffix.lower().lstrip("."),
        file_size=template.source_video.size or 0,
    )


def build_template_file(template: TemplateVideo) -> TemplateVideo:
    generated_dir = Path(settings.MEDIA_ROOT) / "template_center" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = generated_dir / f"template_{template.id}_{template.version}.npz"

    try:
        save_pose_template(
            ref_video=template.source_video.path,
            task_model=settings.FITNESS_ACTION_EVAL["MODEL_PATH"],
            num_poses=1,
            smooth_window=template.smooth_window,
            template_path=str(output_path),
            frame_stride=template.frame_stride,
            preview=False,
        )
        template.template_file_path = str(output_path)
        template.status = TemplateVideo.Status.READY
        template.build_error = ""
        template.save(update_fields=["template_file_path", "status", "build_error", "updated_at"])

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
        template.build_error = str(exc)
        template.save(update_fields=["status", "build_error", "updated_at"])
        raise
    return template


def delete_template_bundle(template: TemplateVideo) -> None:
    source_storage = template.source_video.storage
    source_name = template.source_video.name
    asset_paths = {
        item.file_path
        for item in FileAsset.objects.filter(biz_id=template.id, biz_type__in=[FileAsset.BizType.TEMPLATE_SOURCE, FileAsset.BizType.TEMPLATE_FILE])
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
