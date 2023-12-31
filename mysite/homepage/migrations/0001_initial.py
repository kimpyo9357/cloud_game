# Generated by Django 4.2.2 on 2023-06-10 18:35

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_rule', models.CharField(max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='UserData',
            fields=[
                ('name', models.CharField(max_length=100)),
                ('identifier', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('password', models.CharField(max_length=100)),
                ('nickname', models.CharField(max_length=100)),
                ('check_author', models.BooleanField(default=0)),
                ('pub_date', models.DateTimeField(verbose_name='date published')),
                ('user_rule', models.ForeignKey(db_column='user_rule', on_delete=django.db.models.deletion.CASCADE, to='homepage.user')),
            ],
        ),
    ]
