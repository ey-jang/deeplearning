### 데이터->모델->평가->손실 줄이기(최적화)->다시 모델링->결과
### Prediction/Logit: 각 class 별로 예측한 값
### Loss/Cost : 얼마나 틀렸는지 계산
### Optimization : loss 최소화
### Result : 평가    

   

딥러닝 용어
----------

### CNN(Convolution neuron network)
### Convolution : 합성곱 /이미지와 합성해서 특징을 뽑아냄.
### Weight-학습하려고 하는 대상
### Y = wa + b(bias)
### Pooling layer : 이미지가 가지고 있는 특징을 줄인다.(압축)
### Activation function : 앞에서 특징을 뽑고, 불필요한 음수 부분 없애기(ReLU)
### Softmax : 수치를 유도 /softmax를 거쳐 모든 값 합이 1이 되도록 만든다.
### Loss /cost function : 얼마나 틀렸는지 계산
### Optimization : loss function을 최소로 
### Learning rate : learning rate가 너무 낮아도 높아도 안좋다. / 적정 조절
### Batch Size : 몇 장을 넣을건지 정하는 것
### Epoch : batch size 다음 에폭 수만큼 다시 봐야한다.
### Label/ground truth: 데이터를 받으면 데이터에 대한 정답(레이블)  
       
CNN모델 구조
-----------
### Feature extraction : 특징 추출->fully connected layer에서 결정을 내림
1. Convolution Layer : 특징을 합성
2. Pooling layer(Max Pooling) : 특성을 뽑은 것 중 가장 중요한 것 뽑기(가장 큰 특성 압축)
3. Activation function(ReLU) : 0미만 없애기   

Tensor
------
### np_argmax(arr) : 가장 큰 인덱스의 위치
### np_unique(arr) : 중복 제외하고 유니크한 값   

시각화
------
### import numpy as np
### import matplotlib.pyplot as plt

### %matplotlib inline : 주피터 내부에 그래프를 띄우겠다. 


이미지
------
### 이미지 합치기
<pre>
<code>
Import cv2
dog_image = cv2.resize(image, (275, 183)) #강아지 사이즈 조정
dog_image.shape
out[] ((183, 275, 3), (183, 275, 3))
dog_image.shape, cat_image.shape #강아지랑 고양이랑 사이즈 같다.
</code>
</pre>


### 이미지를 합칠 때 투명도(alpha)를 주면서 이미지 합치기 
