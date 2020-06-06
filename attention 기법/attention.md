Attention 기법
==============

Seq2seq 모델(RNN)
-----------------

![Alt text](attention 기법/img/seq2seq.png)
# seq2seq- 번역 문제 학습, RNN 구조
# 원리 : encoder에 단어를 하나씩 입력, 마지막으로 입력(x2) 받았을 때 출력 -> context
# context를 decoder가 전달 받아 sos(start-of-sequence)가 처음 입력 받아 eos(end-of-sequence)가 나올 때까지 입출력 반복


Tokenizer 과정
--------------
![Alt text](attention 기법/img/tokenizer_e.png)
![Alt text](attention 기법/img/tokenizer_k.png)

# Tokenizer : 단어 -> 숫자 
# Embedding : feature vector가 되도록 한다.

기존의 Seq2seq 모델의 문제점
---------------------------
# encoder의 앞쪽과 decoder의 뒤쪽이 멀리 떨어져 있다. -> 기울기 소실이 일어날 수 있다. 
# sequence 길이가 긴 경우 context에서 병목 현상이 발생해 번역 성능 저하

# 해결 -> Attention Mechanism
# 기본 아이디어 : decoder에서 출력하는 매 시점(time step)마다 인코더에서의 전체 입력 문장을 다시 한번 참고한다

Querying 과정
-------------
# attention 기법/img/querying.png
# 2019 -> Key, EndGame -> Value
# Querying 과정 - 쿼리를 날리면 키를 비교해서 값을 출력

Attention mechanism
--------------------
![Alt text](attention 기법/img/mechanism.png)
# attention mechanism : Q가 쿼리, K가 key, V가 value로 쿼리와 키를 비교하고 유사도인 Comparison을 반영하여 값을 합성

Attention mechanism을 사용한 RNN
--------------------------------
![Alt text](attention 기법/img/rnn attention.png)
# encoder의 hidden layer를 key, value로 사용하고, decoder에서 하나 앞선 time-step인 hidden layer를 query로 사용


Transformer 
-----------
![Alt text](attention 기법/img/transformer.png)
# The Transformer: RNN을 배제하고 Attention만을 이용해 최고의 성능을 끌어낸 연구
# 학습속도가 매우 빠르며 성능도 RNN보다 우수
* Seq2seq와 유사한 transformer 구조
*	Scaled Dot-Product Attention, Multi-Head Attention블록이 알고리즘 핵심
*	병렬 계산 가능
*	입력된 단어의 위치를 표현 위해 Positional Encoding 사용

# seq2seq 모델과의 차이점 
*	Seq2seq - RNN으로 되어 있어 순차적으로 이루어짐.
*	Transformer – 병렬 계산

Transformer - Multi-head Attention - Scaled Dot-Product Attention
------------------------------------------------------------------
![Alt text](attention 기법/img/scaled dot-product.png)
# Query, Key, Value 각각은 병렬로 계산할 수 있어 매트릭스로 바꿔 계산
# Softmax로 유사도를 0에서 1사이 값으로 Normalize하고 유사도 v를 결합해 attention value를 계산

Transformer - Multi-head Attention
----------------------------------
![Alt text](attention 기법/img/multi-head.png)
# Linear연산을 이용해 Q, K, V 차원을 감소시켜 차원을 동일하게 맞추고 Scaled Dot-Product Attention을 h개 모아 병렬적으로 연산

Transformer - Outputs
---------------------
![Alt text](attention 기법/img/transformer2.png)
# Feed-Forward : fully-connected, relu, fully-connected layer로 구성됩니다.
# Add&Norm : Forward-path와 skip connection을 더한 후 layer normalization
# Output Softmax : feature 출력이 나오게 되면 linear연산을 이용해 출력 단어 종류 수에 맞추고, softmax를 이용해 어떤 단어인지 classification문제를 해결
