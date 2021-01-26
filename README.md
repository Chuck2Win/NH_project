# NH 투자증권 project
https://dacon.io/competitions/official/235658/data/  
주어진 기사 데이터를 토대로, 광고성 기사인지 유의미한 정보를 담고 있는 기사인지를 판별하는 알고리즘 개발하기  

## 평가지표  
정확성 (70%) + 속도(30%)  
정확성 지표로는 200여 팀에서 상위 20등에 들었고, 속도는 비공개했기 때문에 상위 몇등인지는 모름.  

## 모델  
### 1. Feature  
feature로는 기사 데이터에서 본문 기사와, 해당 기사의 길이, 그리고 길이가 512이상인지 아닌지의 것만 활용함  

### 2. Train data, Val data, Test data 결과

### 3. 속도 측면

## 환경  
학습은 google colab pro에서 함. GPU : Tesla p100
colab 기본 설치 library 외에 transformers, imbalanced-learn, kobert-transformers를 설치  
requirements.txt(전체 colab에 있는 library 외에 위 3개 library가 추가됨)  
colab을 활용하시는 분은 아래 코드만 기입해서 실시하면 됨.
! pip3 install transformers
! pip3 install imbalanced-learn
! pip3 install kobert-transformers

## 경량화 시킨 모델

## 코드 실행
### 데이터 전처리
### 데이터 학습
### Inference

