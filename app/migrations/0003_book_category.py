# Generated by Django 5.0.6 on 2024-07-01 07:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='book',
            name='category',
            field=models.TextField(blank=True, null=True),
        ),
    ]