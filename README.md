## 해당 데이터는 파기 완료했습니다.

# NH 투자증권 project
https://dacon.io/competitions/official/235658/data/  
주어진 기사 데이터를 토대로, 광고성 기사인지 유의미한 정보를 담고 있는 기사인지를 판별하는 알고리즘 개발하기  

# Data Description
![data](https://github.com/Chuck2Win/NH_project/blob/main/result/data.png) 
(해당 데이터는 데이콘에 남아있는 예시 파일임. 문제가 될 시 삭제하겠음)  
n_id : id, date : 날짜, title : 제목, content : 본문 내용, ord : 해당 날짜의 발행된 기사 순서, info : 0이면 뉴스 1이면 광고  

![data](https://github.com/Chuck2Win/NH_project/blob/main/result/data2.png)    
![data](https://github.com/Chuck2Win/NH_project/blob/main/result/data3.png)  
광고보다 뉴스의 수가 더 많고, 광고의 길이가 평균적으로 뉴스보다 짧음.  
광고의 경우 중복(content기준)되는 비율이 매우 높음.  


## 평가지표  
정확성 (70%) + 속도(30%)  
상위 20등.  

## 모델    
![model](https://github.com/Chuck2Win/NH_project/blob/main/result/kobertmodel.jpg)  
### 1. Feature  
feature로는 기사 데이터에서 본문 기사와, 해당 기사의 길이, 그리고 길이가 512이상인지 아닌지의 것만 활용함  

### 2. Train, Val, Test data 결과
![train val test](https://github.com/Chuck2Win/NH_project/blob/main/result/train_val_test.png)
### 3. 속도 측면  
![train val test](https://github.com/Chuck2Win/NH_project/blob/main/result/speed.png)

## 환경  
### 데이터는 파기함, 데이터가 있는 현업자의 경우, argparser에 해당 위치로 변경해서 진행하면 됨, code만 참고하시면 될 것 같습니다.
학습은 google colab pro에서 함. GPU : Tesla p100  
colab 기본 설치 library 외에 transformers, imbalanced-learn, kobert-transformers, sentencepiece를 설치  
requirements.txt(전체 colab에 있는 library 외에 위 4개 library가 추가됨)  
colab을 활용하는 경우 아래 코드만 기입해서 실시하면 됨.  
! pip3 install transformers  
! pip3 install imbalanced-learn  
! pip3 install kobert-transformers  
! pip install sentencepiece

requirements.txt로 설치하는 경우  
pip install -r requirements.txt

## 경량화 시킨 모델  
teacher를 원래 model, student를 lstm를 이용한 분류기  
![model](https://github.com/Chuck2Win/NH_project/blob/main/result/distill.jpg)  

### Train, Val, Test data 결과
![model](https://github.com/Chuck2Win/NH_project/blob/main/result/distilltrainvaltest.png)  

### 속도 측면
![model](https://github.com/Chuck2Win/NH_project/blob/main/result/distillspeed.png)  

### 정리  
정확성은 떨어졌으나, 속도가 매우 빨라졌음(32배 증가)    
Total parameter : 기존 (92,188,424) -> 경량(1,288,968) (1.3% 수준)     

## 코드 실행    
### 각각의 실행 파일에 들어가서 보시면, 바꿀 수 있는 hyper parameter들이 있습니다.  
git clone https://github.com/Chuck2Win/NH_project.git  
cd NH_project  
pip install -r requirements.txt  
python3 preprocess.py # preprocessing(train test split, over sampling 도 포함)  
python3 train.py # train  
python3 inference.py --input_data = (input data 위치) --model = (model의 위치) --result = (결과를 저장할 위치) #inference  

## 코드 그 외
### 1. [TEST].ipynb  
해당 파일은 저장된 모델을 불러와서, 데이터를 전처리하고 데이터를 예측하는 과정을 담은 파일임.  
### 2. 데이콘][BERT][중복제거없이][oversampling][distill].ipynb
해당 파일은 STUDENT MODEL을 학습시키는 과정임, 이 때 데이터에는 Teacher Model의 Logit 값이 있어야함.  
