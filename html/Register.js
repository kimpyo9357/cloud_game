// 가입부분 체크

function signUpCheck(){

//   let email = document.getElementById("email").value
  let user_name = document.getElementById("name").value
  let user_pw = document.getElementById("password").value
  let passwordCheck = document.getElementById("passwordCheck").value
  let user_dept = document.getElementById("dept").value
  let user_stn = document.getElementById("stn").value
  let check = true;

  // 이름확인
  if(user_name===""){
    document.getElementById("nameError").innerHTML="이름이 올바르지 않습니다."
    check = false
  }else{
    document.getElementById("nameError").innerHTML=""
  }


  // 비밀번호 확인
  if(user_pw !== passwordCheck){
    document.getElementById("passwordError").innerHTML=""
    document.getElementById("passwordCheckError").innerHTML="비밀번호가 동일하지 않습니다."
    check = false
  }else{
    document.getElementById("passwordError").innerHTML=""
    document.getElementById("passwordCheckError").innerHTML=""
  }

  if(user_pw===""){
    document.getElementById("passwordError").innerHTML="비밀번호를 입력해주세요."
    check = false
  }else{

  }
  if(passwordCheck===""){
    document.getElementById("passwordCheckError").innerHTML="비밀번호를 다시 입력해주세요."
    check = false
  }else{

  }
  if(user_dept===""){
    document.getElementById("deptError").innerHTML="학과명이 올바르지 않습니다."
    check = false
  }else{
    document.getElementById("deptError").innerHTML=""
  }
  if(user_stn===""){
    document.getElementById("stnError").innerHTML="학번이 올바르지 않습니다."
    check = false
  }else{
    document.getElementById("stnError").innerHTML=""
  }
  if(check){
    document.getElementById("deptError").innerHTML=""
    document.getElementById("stnError").innerHTML=""
    document.getElementById("nameError").innerHTML=""
    document.getElementById("passwordError").innerHTML=""
    document.getElementById("passwordCheckError").innerHTML=""
    
    //비동기 처리이벤트
    setTimeout(function() {
      alert("가입이 완료되었습니다.")
      window.location='Login.html';
  },0);
  }
}