# Profile-Database (Team. P.H.)
## <학생들의 프로필을 관리할 수 있는 데이터베이스 구축>
### 제안 배경 및 목표 설명
- 많은 사이트에서 본인인증을 하기 위한 수단으로 ID/PW를 사용한다.
- 본 프로젝트는 인증절차를 구현하는 방법을 학습한다.
- 프로젝트를 구성함에 있어 필요한 홈페이지 구성, 서버관리를 학습한다.
- 프로젝트를 통한 git과 GitHub를 사용한 다른 사람과의 협력 방법을 학습한다.
### 시스템 기능
- <strong>서버</strong> : 리눅스에서 서버를 개발해 학생들의 정보를 입력하여 홈페이지에서 로그인을 할 경우 관리자에게 요청 후 관리자가 데이털르 확인 후 로그인 처리
- <strong>관리자</strong> : 학생들의 정보 및 데이터를 가지고 있으며 학생들이 로그인 할 때 데이터를 확인 후 로그인 허용
- <strong>사용자</strong> : 사용자는 학생들을 대상으로 로그인 화면에서 본인의 정보를 입력하면 관리자에게 권한 요청. 사용자들은 본인의 프로필 수정 권한을 가짐
### 시스템에서 제공하는 기능을 상세하고 정확하게 서술
- 홈페이지에 접속 시 Login 페이지에 접속한다. 
- Database Query에 입력이 되어 있는 ID/PW로 접근 시 User의 경우 Info Page, Admin의 경우 Setup Page로 접근한다. 
- 만약 Query에 입력되지 않은 ID/PW로 접근 시 다른 Page의 접근을 허가하지 않는다. 
- Query에 입력되지 않은 사용자가 ID 생성을 원할 시 ID Require을 이용해 해당 내용을 기술, 기술한 내용은 Admin에게 Message로 남게 된다. 
- Admin은 해당 정보를 확인 후 Query에 입력을 허락/거부를 결정한다. User의 경우 Info Page로 접근 시 기본적인 정보는 수정 불가인 상태로 표시된다. 
- 만일 User가 추가 개인 정보의 입력을 원할 시 User Page의 ‘+’ 버튼을 눌러 추가 정보를 기록 하여 Database에 저장 가능하다. 
- Admin의 경우 Setup Page로 접근한다. Setup Page에서 Database Query에 기록된 데이터의 수정, 삭제 권한을 가지고, ID Require Message를 판단 가능하다.

![그림1](https://user-images.githubusercontent.com/71916473/168409231-a4d9bd1b-199b-43e6-ae4b-f383610c4a68.png)
#### User Interface (가정)
![그림2](https://user-images.githubusercontent.com/71916473/168409259-2e524531-bf17-4d28-afea-ad6302d04938.png)
![그림3](https://user-images.githubusercontent.com/71916473/168409261-e848409c-adc8-48cd-a7d7-e997baa1eb37.png)
![그림4](https://user-images.githubusercontent.com/71916473/168409262-c505947a-78f3-42f8-8d44-41a1c2cc2e4a.png)

### 개발 일정 및 역할 분담
#### 개발 일정
- 5/10 ~ 5/16: 계획 설립 및 세부 사항 결정 & 리눅스 서버를 이용한 임시 홈페이지 생성
- 5/17 ~ 5/23: 홈페이지를 html을 이용한 제작
- 5/24 ~ 5/30: Query 와 작동 코드 생성
- 5/31 ~ 6/6: 미흡한 부분 수정 및 오류 확인
- 6/7 ~: 미흡한 부분 수정 및 추가 구현
#### 역할 분담
- 표세훈 (팀장) : 리눅스 서버 관리/구현 & 홈페이지 코드 담당 / E-mail : kimpyo9357@navercom/ GitHub ID : kimpyo9357
- 안상현 (팀원) : 홈페이지 제작 담당 /E-mail : tedan96@naver.com/ GitHub ID : tedan96
### GitHub Link
- https://github.com/kimpyo9357/Profile-Database
### 저작권
- 해당 프로그램은 <strong>GPL v3.0</strong>을 따르며 <strong>Copyright 2022. Team P.H.</strong> 에게 있습니다.
### 참고 문헌
- 생활코딩! HTML+CSS+자바스크립트: 이고잉 지음/위키북스 기획·편집
