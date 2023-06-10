from django.db import models

class User(models.Model):
    user_rule = models.CharField(max_length=200)
    def __str__(self):
        return self.user_rule

class UserData(models.Model):
    user_rule = models.ForeignKey(User, on_delete=models.CASCADE, db_column='user_rule')
    #user_id = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    identifier = models.CharField(max_length=100, primary_key=True)
    password = models.CharField(max_length=100)
    nickname = models.CharField(max_length=100)
    check_author = models.BooleanField(default=0)
    pub_date = models.DateTimeField('date published')
    def __str__(self):
        return self.name

