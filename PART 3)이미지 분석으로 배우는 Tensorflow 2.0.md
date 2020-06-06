part 3 이미지 분석으로 배우는 tensorflow
=======================================

### Augmentation : 여러 환경에서 적응 가능하도록 트레이닝
### callbacks : epoch or step 단위로 이벤트를 일으키는 옵션, 정해진 시간대에 running rate를 준다.
### 모델 저장 및 불러오기
<pre><code>from glob import glob # 외부 파일 불러오기
os.getcwd() # 현재 경로
os.listdir()
os.listdir(‘dataset/mnist_png/training/0/’) # 폴더 경로
glob(dataset/mnist_png/training/0/*.png’) # 경로가 포함된 모든 파일
data_paths[-1] #마지막에 있는 것 가져오기
path = data_paths[0] 
path # 첫번째에 있는 것	경로 불러오기
</code></pre>
### 데이터 분석(MNIST)
<pre><code>os.listdir(‘dataset/mnist_png/training/’) # 데이터 확인</code></pre>
### 데이터별 개수 비교
<pre><code>nums_dataset = []
for lbl_n in label_nums:
	data_per_class = os.listdir(‘dataset/mnist_png/training/’ + lbl_n) #각 레이블별 데이터, 데이터 셋을 클래스 별로 확인
image_pil = Image.open(path) #Pillow로 열기
image = np.array(image_pil)
TensorFlow 로 열기
gfile = tf.io.read_file(path)
image = tf.io.decode_image(gfile)
TensorShape([28, 28, 1])	#1이 채널
path
path.split(‘\\’) # \기준으로 쪼개짐
int(label) # 숫자로 변환
</code></pre>
### 데이터 이미지 사이즈 알기
<pre><code>from tqdm import tqdm_notebook
heights = []
widths = []
len(data_paths[:10])
heights = []
widths = []

for path in tqdm_notebook(data_paths):
    image_pil = Image.open(path)
    image = np.array(image_pil)
    h, w = image.shape
    
    heights.append(h)
    widths.append(w)	
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.hist(heights)
plt.title(‘Heights’)
plt.axvline(np.mean(heights), color = ‘r’, linestyle=’dashed’, linewidth=2) # 평균값

plt.subplot(122)
plt.hist(widths)
plt.title(‘Widths’)
plt.show()
</code></pre>
### 이미지를 배치 사이즈만큼 잘라서 넣어주기
#### 배치 사이즈만큼 모델에 넣기
### 배치가 다 돌아가면 한 에폭

<pre><code>Images in List
batch_image = []
for path in data_paths[:8]: #데이터패스중 8개만 받기
    image = read_image(path)
    batch_image.append(image) 
plt.imshow(batch_images[0])
plt.show()
batch = tf.convert_to_tensor(batch_images)
batch.shape
(batch_size, height, width, channel)
### 데이터가 4차원
def make_batch(batch_paths):
	batch_images = []
	for path in batch_paths:
	image = read_image(path)
	batch_images.append(Image)

	return tf.convert_to_tensor(batch_images_

batch_size = 16;

for step in range[4]:
	batch_images = make_batch(data_paths[step*batch_size : (step + 1) * batch_size])
	plt imshow(batch_images[0])
	plt.show() # 시각화
batch_images.shape

data generator # 데이터 모델링 간편하게
data_paths = os.listdir(‘dataset/mnist_png/0/*.png’)
data_paths[0]
Load Image
gfile = tf.io.read_file(path)
image = tf.io.decade_image(gfile)	#이미지 열기
image.shape
plt.imshow(image[:, :, 0], ‘gray’)
plt.show()
Set Data Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(		#데이터에 변환을 주면서 이미지 학습
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
image.shape
inputs = image[tf.newaxis, …] 	#차원 수 늘리기
inputs.shape
image = next(iter(datagen.flow(inputs)))
image.shape
</code></pre>

### Transformation 
#### data generator - 변환 주기
<pre><code>datagen = ImageDataGenerator(
	width_shift_range=0.3 #0.3만큼 랜덤하게 변환
	zom_range = 0.3 #위로 옮겨지거나 아래로 옮겨지거나
)
outputs = next(iter(datagen.flow(inputs)))</code></pre>

### rescale - train, test 모두 한다.
<pre><code>train_datagen = ImageDataGenerator(
	zoom_range = 0.7, #train에만 해준다.
	rescale = 1./255.)
test_datagen = ImageDataGenerator(
    rescale=1./255
)</code></pre>

### Preprocess
<pre><code>train_datagen = ImageDataGenerator(
	rescale = 1./255.,
	width_shift_range = 0.3,
	zoom_range = 0.2,
	horizontal_flip = True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],	#채널 빼고 2개
        batch_size=batch_size,
        color_mode='grayscale',
	class_mode = ‘categorical’
)
validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        color_mode='grayscale'
	class_mode = ‘categorical’
)</code></pre>

### Training
<pre><code>model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=num_epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator))</code></pre>

os.path.basename(path) #파일명만 가져오기
os.path.exist(path) #path가 있는지 없는지

### class 수 확인
<pre><code>classes_name = []
for path in train_paths:
	cls_name = get_class_name(path)
	class_names.append(cls_name)
class_names = [get_class_name(path) for path in train_paths]
unique_classes = np.unique(classes_name, return_counts=True) #클래스가 몇 개 있는지 (counts)
unique_classes
plt.bar(*unique_classes)
plt.xticks(rotation=45)
plt.show()</code></pre>

### DataFrame 생성
<pre><code>data_ex = {'a':[1, 2, 3], 'b':[10, 20, 30], 'c':[100, 200, 300]}

df_ex = pd.DataFrame(data_ex)
df_ex
data = {'path': train_paths, 'class_name': classes_name}
df = pd.DataFrame(data)
df.head()</code></pre>

### 만들어진 DataFrame 저장
<pre><code>train_csv_path = ‘train_dataset.csv’
test_csv_path = ‘test_dataset.csv’</code></pre>

### dataframe 이용해서 학습하기
<pre><code> import pandas as pd
train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')
train_df.head()
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.3,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='path',
        y_col='class_name',
        target_size=input_shape[:2],
        batch_size=batch_size
)
validation_generator = test_datagen.flow_from_dataframe(
        test_df,
        x_col='path',
        y_col='class_name',
        target_size=input_shape[:2],
        batch_size=batch_size
)
tf.data
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_image(gfile, dtype=tf.float32)
    return image
dataset = tf.data.Dataset.from_tensor_slices(train_paths)
dataset = dataset.map(read_image, num_parallel_calls=AUTOTUNE)</code></pre>

### 배치로 묶어서 모델에 넣어주기
<pre><code>
dataset = tf.data.Dataset.from_tensor_slices(train_paths)
dataset = dataset.map(read_image)
dataset = dataset.batch(4) #batch_size 4만큼 묶인다.</code></pre>

### Shuffle(섞기) Label하고 같이 넣기
<pre><code>dataset = tf.data.Dataset.from_tensor_slices((train_paths, labels))
dataset = dataset.map(load_data, num_parallel_calls=AUTOTUNE)
dataset = dataset.batch(4)
dataset = dataset.shuffle(buffer_size=len(train_paths))
dataset = dataset.repeat() # 반복적으로 돌아갈 수 있도록</code></pre>
### tensorflow 함수로 label얻기
<pre><code>
def onehot_encoding(label):
	return np.array(class_names == label, np.uint8)
Data Preprocess
train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
train_dataset = train_dataset.map(load_image_label, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(image_preprocess, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.shuffle(buffer_size=len(train_paths))
train_dataset = train_dataset.repeat()
test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
test_dataset = test_dataset.map(load_image_label, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.repeat()
</code></pre>

### training
<pre><code>steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(test_paths) // batch_size
model.fit_generator(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_dataset,
    validation_steps=validation_steps,
    epochs=num_epochs
)</code></pre>

### callbacks:학습 도중 이벤트
#### tensorboard열기
<pre><code>callbacks 
logdir = os.path.join('logs',  datetime.now().strftime("%Y%m%d-%H%M%S"))
 tf.keras.callbacks.TensorBoard(
	log_dir = logdir,
	write_graph = True,
	write_images = True,
	histogram_freq = 1
)</code></pre>
%tensorboard --logdir logs --port 8008
Training
LamdaCallback : 맞춤형 그래프 가져오기(복붙)
### Define the per-epoch callback.
<pre><code>cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)</code></pre>


### callbacks-learing rate schedule : 최적화된 위치까지 도달하게 하는 것
<pre><code>def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * math.exp(0.1 * (10 - epoch))

learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)</code></pre>

### checkpoint : 모델이 학습하다가 weight를 저장시킴. 나중에 weight로 돌아갈 수 있도록
<pre><code> save_path = 'checkpoints'
checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# val_accuracy가 올라가면 저장하고, 아니면 저장안함.(save_best_only=True)
# val_accuracy이면 mode = ‘max’ , loss이면 ‘min’</code></pre>

### history 들여다보기
<pre><code>history.history.keys()
history.params
new_model = history.model
plt.plot(history.history[‘accuracy’])
plt.plot(history.history[‘val_accuracy’])
plt.title(“Model Accuracy”)
plt.ylabel(‘accuracy’)
plt.xlabel(‘epoch’)
plt.legend([‘tran’, ‘validation’])
plt.show()
plt.plot(history.history[‘loss’])
plt.plot(history.history[‘val_loss’])
plt.title(‘Model Loss’)
plt.ylabel(‘loss’)
plt.xlabel(‘epoch’)
plt.legend([‘tran’, ‘validation’])
plt.show()</pre></code>

### 이미지를 load 직접load해서 넣는 방법
<pre><code>
path = train_paths[0]
test_image, test_label = load_image_label(path)
test_image.shape
test_image = test_image[tf.newaxis, ...]
test_image.shape
pred = model.predict(test_image)
pred
generator에서 데이터 가져오는 방법
generator에 넣는 방법
pred = model.predict_generator(test_dataset.take(1)) #한 배치만 가져온다
evals = model.evaluate(image, label)</code></pre>

### tf케라스로 저장하기
<pre><code>
save_path=’my_model.h5’
model.save(save_path, include_optimizer=True)
tf.keras.models.load_moel(‘my_model.h5’) #모델 불러오기
</code></pre>
### 모델을 weight만 저장하기
<pre><code>model.save_weights(‘model_weights.h3)
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
from tensorflow.keras.models import model_from_json
with open('model_architecture.json', 'r') as f:
    model = model_from_json(f.read())
model.load_weights('model_weights.h5')
</code></pre>
