2-02 TensorFlow 기초 사용법
===========================   

### Tensor 생성 list -> Tensor / array->Tensor
<pre>
<code>tf.constant([1, 2, 3])
</code></pre>
### 데이터 타입 적용 
<pre><code>tf.constant([1, 2, 3],dtype=tf.float32)</pre></code>

### 데이터 타입 변환 
<pre><code> tf.cast(tensor, dtype = tf.uInt8)
 tf.random.normal([3, 3])
 tf.random.uniform([4, 4])
</code></pre>
### Data Preprocess(MNIST)
<pre><code>from tensorflow.keras import datasets
mnist = datasets.mnist
(tranin_x, train_y), (test_x, test_y) = mnist.load_data()
image = train_x[0] //데이터 하나만 뽑기
image.shape 
plt.imshow(image, ‘gray’) //시각화
plt.show()
</code></pre>
### Channel 관련
#### 차원수 늘리기
<pre><code> new_train_x = np.expand_dims(train_x, -1) # 맨뒤에 1붙이기
 new_train_x.shape
 disp = new_train_x[0, :, :, 0] / disp = np.squeeze(new_train_x[0])
 disp.shape	#결과 : (28, 28) shape를 줄이기
</code></pre>   

### 원 핫 인코딩 : 컴퓨터가 이해할 수 있는 형태로 변환해서 label을 주도록 함.
### kernel_size: filter(weight)의 사이즈
### strides : 몇 개의 pixel을 skip하면서 훑어 지날지
### padding : zero padding만들건지 ,VALID 는 padding 없고, SAME은 padding 있다.
### activation: activation function만들건지
## Visualization
### pooling : 이미지가 줄어든다. 
<pre><code>tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='SAME')
</code></pre>

## Optimization & Training(Beginner)(모델 학습)
<pre><code>from tensorflow.keras import layers
from tensorflow.keras import datasets
</code></pre>
### crossentropy : 2개 -> binary_crossentropy / 2개 이상 : categorical_crossentropy
### 원핫인코딩을 주지 않았을 때 : sparse_categorical_crossentropy
### 원핫인코딩을 주었을 때 : categorical_crossentropy
### Compile – Optimizer 적용   


   
## 모델 평가 : accuracy를 이름으로 넣기

### Training : 학습용 Hyperparameter 설정 
### epoch ->데이터를 하나씩 보는데 다 보면 한 epoch
### batch_size만큼 한 모델에 넣어줘야함.(메모리의 효율 위해)
<pre><code>model.fit(train_x, train_y, 
          batch_size=batch_size, 
          shuffle=True, 
          epochs=num_epochs)
</code></pre>
 

