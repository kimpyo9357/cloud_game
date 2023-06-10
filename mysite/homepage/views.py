from django.shortcuts import render, redirect
from .models import  User, UserData
from django.db.models import Q
import datetime

def login(request):
    if request.method == 'GET':
        return render(request, 'homepage/Login.html');
    elif request.method == 'POST':
        check_id = request.POST['user_stn']
        check_pw = request.POST['user_pw']
        if (UserData.objects.filter(Q(identifier=check_id) & Q(password=check_pw) & Q(check_author=1)).exists()):
            data = UserData.objects.filter(identifier=check_id).values()
            return render(request, 'homepage/Information.html', {'data' : data})
        else:
            return render(request, 'homepage/Login.html', {
                'error_message' : "잘못된 ID 혹은 Password를 입력하였습니다."
            })

def author(request):
    data = UserData.objects.filter(check_author=1)
    return render(request, 'homepage/Author.html', {'data' : data});
    
def authorlogin(request):
    if request.method == 'GET':
        return render(request, 'homepage/AuthorLogin.html');
    elif request.method == 'POST':
        check_id = request.POST['user_stn']
        check_pw = request.POST['user_pw']
        if (UserData.objects.filter(Q(identifier=check_id) & Q(password=check_pw) & Q(user_rule_id=2)& Q(check_author=1)).exists()):
            data = UserData.objects.get(identifier=check_id)
            return redirect('homepage:author')
        else:
            return render(request, 'homepage/AuthorLogin.html', {
                'error_message' : "잘못된 ID 혹은 Password를 입력하였습니다."
            })
    
def information(request):
    return render(request, 'homepage/Information.html');
    
def message(request):
    if request.method == 'GET':
        data = UserData.objects.filter(check_author=0)
        return render(request, 'homepage/Message.html', {'data' : data})
    elif request.method == 'POST':
        find_id = request.POST.getlist('check')
        if len(find_id)!=0 :
            for i in find_id:
                UserData.objects.filter(identifier=i).update(check_author=1)
        return redirect('homepage:author')

def delete(request):
    if request.method == 'GET':
        data = UserData.objects.filter(check_author=1)
        return render(request, 'homepage/Delete.html', {'data' : data})
    elif request.method == 'POST':
        find_id = request.POST.getlist('check')
        if len(find_id)!= 0 :
            for i in find_id:
                UserData.objects.filter(identifier=i).delete()
        return redirect('homepage:author')


def signup(request):
    data = User.objects.all()
    if request.method == 'GET':
        return render(request, 'homepage/SignUp.html', {'data' : data})
    elif request.method == 'POST':
        input_name = request.POST['name']
        input_id = request.POST['id']
        input_pw = request.POST['password']
        input_pw2 = request.POST['password2']
        input_rule = request.POST['rule']
        input_nickname = request.POST['nickname']
        #input_stn = request.POST['stn']
        print(input_name,input_id,input_pw,input_rule)
        if len(input_name) == 0 or len(input_id) == 0 or len(input_pw) == 0 or len(input_rule) == 0 or len(input_nickname) == 0 :
            return render(request, 'homepage/SignUp.html', {'data' : data, 'message' : 1})
        elif input_pw != input_pw2:
            return render(request, 'homepage/SignUp.html', {'data' : data, 'message' : 2})
        else:
            UserData.objects.create(user_rule_id=input_rule,name=input_name,identifier=input_id,password=input_pw,nickname=input_nickname,check_author=0,pub_date=datetime.datetime.now())
            return render(request, 'homepage/SignUp.html', {'data' : data, 'message' : 0})
            #return redirect('homepage:login')
        
        '''if len(input_name) == 0 or len(input_id) == 0 or len(input_pw) == 0 or len(input_rule) == 0 or len(input_dept) == 0 or len(input_stn) == 0:
            return render(request, 'homepage/SignUp.html', {'data' : data, 'message' : 1})
        elif input_pw != input_pw2:
            return render(request, 'homepage/SignUp.html', {'data' : data, 'message' : 2})
        elif len(UserData.objects.filter(user_id=input_stn)) != 0:
            print(len(UserData.objects.filter(user_id=input_stn)))
            return render(request, 'homepage/SignUp.html', {'data' : data, 'message' : 3})
        else:
            UserData.objects.create(user_rule_id=input_rule,user_id=input_stn,name=input_name,identifier=input_id,password=input_pw,department=input_dept,check_author=0,pub_date=datetime.datetime.now())
            return render(request, 'homepage/SignUp.html', {'data' : data, 'message' : 0})
            #return redirect('homepage:login')'''
