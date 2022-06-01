from django.urls import path

from . import views

app_name = "homepage"
urlpatterns = [
    path('', views.LoginView.as_view(), name='Login'),
    path('Author/', views.AuthorView.as_view(), name='Author'),
    path('AuthorLogin/', views.AuthorLoginView.as_view(), name='AuthorLogin'),
    path('Login/', views.LoginView.as_view(), name='Login'),
    path('Message/', views.MessageView.as_view(), name='Message'),
    path('Information/', views.InformationView.as_view(), name='Information'),
    path('SignUp/', views.SignUpView.as_view(), name='SignUp'),
]