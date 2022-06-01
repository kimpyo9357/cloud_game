from django.shortcuts import render
from django.http import HttpResponse
from django.views import generic
from .models import  User
from django.utils import timezone

class AuthorView(generic.ListView):
    model = User
    template_name = 'homepage/Author.html'

class AuthorLoginView(generic.ListView):
    model = User
    template_name = 'homepage/AuthorLogin.html'
    
class MessageView(generic.ListView):
    model = User
    template_name = 'homepage/Message.html'

class InformationView(generic.ListView):
    model = User
    template_name = 'homepage/Information.html'

class SignUpView(generic.ListView):
    model = User
    template_name = 'homepage/SignUp.html'
    

class LoginView(generic.ListView):
    template_name = 'homepage/Login.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        """
        Return the last five published questions (not including those set to be
        published in the future).
        """
        return User.objects.filter(
            pub_date__lte=timezone.now()
        ).order_by('-pub_date')[:5]

    def get_queryset(self):
        """
        Excludes any questions that aren't published yet.
        """
        return User.objects.filter(pub_date__lte=timezone.now())

