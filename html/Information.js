let data = {
    "id" : "test",
  "name" : "안상현",
  "password" : "123456",
  "dept" : "소프트웨어학과", 
  "stn" : "32177384"
};



let Name = document.getElementById("name")
let ID = document.getElementById("id")
let PW = document.getElementById("password")
let Dept = document.getElementById("dept")
let Stn = document.getElementById("stn")

Name.innerText += data.name
ID.innerText += data.id
PW.innerText += data.password
Dept.innerText += data.dept
Stn.innerText += data.stn