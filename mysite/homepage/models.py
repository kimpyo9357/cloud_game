from django.db import models

class User(models.Model):
    user_id = models.IntegerField(default=0)
    pub_date = models.DateTimeField('date published')

class UserData(models.Model):
    user_id = models.ForeignKey(User, on_delete=models.CASCADE)
    identifier = models.CharField(max_length=200)
    password = models.CharField(max_length=200)
    name = models.CharField(max_length=200);
    department = models.CharField(max_length=200);

