from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("evaluation_center", "0002_evaluationhint_evaluationphaseresult"),
    ]

    operations = [
        migrations.AddField(
            model_name="evaluationtask",
            name="progress_percent",
            field=models.PositiveIntegerField(default=0, verbose_name="进度百分比"),
        ),
        migrations.AddField(
            model_name="evaluationtask",
            name="progress_text",
            field=models.CharField(blank=True, default="", max_length=100, verbose_name="进度文本"),
        ),
    ]
