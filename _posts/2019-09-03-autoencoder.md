---
layout: post
title: Autoencoder
excerpt: "tensorflow로 간단한 Autoencoder를 구현해보자"
categories: [GAN practice]
comments: true
---

실습 코드는 Smart Design Lab의 튜토리얼을 참고하였습니다. Auto encoder에 관한 자세한 내용은 아래 블로그 글을 참고하였습니다.
{: .notice}
 
 > [Smart Design Lab](http://www.smartdesignlab.org/dl.html)
 
 > [Untitled](https://untitledtblog.tistory.com/92)
 
 
![aouto-1]({{ site.url }}/img/autoencoder-4.PNG)

Autoencoder는 이미지 데이터의 압축을 위해 연구된 인공신경망 (Artificial Neural Networks, ANNs)입니다. Autoencoder의 구조는 일반적인 feedforward neural networks (FNNs)와 유사하지만, autoencoder는 비지도 학습 (unsupervised learning) 모델입니다. Autoencoder는 기존에 대부분 데이터의 압축을 위해 활용되었으나, 최근에는 딥 러닝 (deep learning)에 대한 연구가 활발해지면서 입력 벡터의 차원 축소, 은닉층의 학습 등에 많이 이용되고 있습니다.

## 구조

![auto-2]({{ site.url }}/img/autoencoder-3.PNG)
Autoencoder의 구조는 일반적인 FNN(Feadforward neural network) 구조와 매우 유사합니다. 다만 입력층(input layer)와 출력층(output layer)의 크기가 항상 같다는 특징이 있습니다. 아래의 그림은 은닉층(hidden layer)이 한개인 autoencoder의 구조를 보여줍니다. 입력층과 출력층의 크기는 같으며 입력층과 은닉층 구간을 `encoder`, 은닉층과 출력층 구간을 `decoder`라고 합니다.

Autoencoder의 중요한 동작은 입력벡터의 차원을 축소하는 것입니다. 이러한 동작은 입력층-은닉층 구간인 encoder에서 수행되며 차원이 축소된 입력 벡터를 code 또는 latent variables라고 합니다. Encoder 영역에서 생성된 code는 은닉층-출력층 구간인 decoder를 거쳐 입력층의 차원과 동일한 차원의 출력벡터로 변환됩니다. 

## 학습 알고리즘

## 구현

tensorflow를 이용해서 간단한 Autoencoder을 구현해보도록 합니다.

### Library import

먼저 실습에 필요한 라이브러리를 가져옵니다. tensorflow, numpy, matplotlib를 사용합니다. 또한 tensorflow에서 제공하는 mnist 데이터도 함께 불러오겠습니다. 

{% highlight python %} 

#텐서플로, numpy, matplotlib의 라이브러리 임포트
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#MNIST 모듈 임포트
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

{% endhighlight %}

### Set parameter

하이퍼파라미터로 사용할 옵션들을 정합니다. 

{% highlight python %} 
learning_rate = 0.01 #최적화 함수에서 사용할 학습률
training_epoch = 20  #전체 데이터를 학습할 총 횟수
batch_size = 100     #미니배치로 한번에 학습할 데이터(이미지)의 갯수

n_input = 28*28      #이미지 크기 28*28 = 784
n_hidden = 256       #은닉층의 뉴런 개수
n_output = 28*28     #input 이미지의 크기와 동일 
{% endhighlight %}

### Model development

{% highlight python %} 

# X라는 플레이스 홀더를 설정
X = tf.placeholder(tf.float32, [None, n_input])

# ENCODER 인코더, n_hidden개의 뉴런을 가진 은닉층 만듬
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))  #가중치 
b_encode = tf.Variable(tf.random_normal([n_hidden])) #편향변수
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X,W_encode),b_encode)) #sigmoid 활성화 함수 적용

# DECODER 디코더, n_output개 이미지 출력
W_decode = tf.Variable(tf.random_normal([n_hidden, n_output])) #가중치
b_decode = tf.Variable(tf.random_normal([n_output])) #편향변수 
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder,W_decode),b_decode)) # 입력값을 은닉층의 크기로, 출력값을 입력층의 크기로 
{% endhighlight %}

### Model training

#### loss function

* $$\frac{1}{m} \sum^{m}_{i} (x_{i}-y_{i})^2$$

{% highlight python %} 
#손실함수(두 값의 거리차이) = X(평가하기 위한 실측값) - 디코더의 결과값
cost = tf.reduce_mean(tf.pow(X-decoder,2))

#최적화 함수 RMSPropOptimizer로 cost를 최소화 함
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#학습진행
init = tf.global_variables_initializer() #변수 초기화
sess = tf.Session() # Session 오픈
sess.run(init) # 텐서플로우로 변수들 초기화 완료(학습 진행 준비 완료)

total_batch = int(mnist.train.num_examples / batch_size) #배치 갯수

for epoch in range(training_epoch): #train 테이터 셋으로 부터 전체 배치를 불러옴
    total_cost = 0
    
    for i in range(total_batch): #모든 배치들에 대하여 최적화 수행
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 배치사이즈에 맞게 x, y값 생성
        _, cost_val = sess.run([optimizer, cost], feed_dict={X:batch_xs}) # X값(이미지데이터)를 통해 최적화 진행
        total_cost += cost_val 
        
    print('Epoct:', '%04d' % (epoch + 1), 'Avg.cost = ', '{:.4f}'.format(total_cost/total_batch)) # Epoct 별 cost 보여줌

print('최적화 완료!')

{% endhighlight %}

출력 결과는 다음과 같습니다.

```python
Epoch: 0001 Avg.cost =  0.2053
Epoch: 0002 Avg.cost =  0.0570
Epoch: 0003 Avg.cost =  0.0470
Epoch: 0004 Avg.cost =  0.0422
Epoch: 0005 Avg.cost =  0.0395
Epoch: 0006 Avg.cost =  0.0382
Epoch: 0007 Avg.cost =  0.0367
Epoch: 0008 Avg.cost =  0.0359
Epoch: 0009 Avg.cost =  0.0351
Epoch: 0010 Avg.cost =  0.0329
Epoch: 0011 Avg.cost =  0.0320
Epoch: 0012 Avg.cost =  0.0314
Epoch: 0013 Avg.cost =  0.0311
Epoch: 0014 Avg.cost =  0.0307
Epoch: 0015 Avg.cost =  0.0301
Epoch: 0016 Avg.cost =  0.0294
Epoch: 0017 Avg.cost =  0.0291
Epoch: 0018 Avg.cost =  0.0288
Epoch: 0019 Avg.cost =  0.0284
Epoch: 0020 Avg.cost =  0.0279
최적화 완료!
```
### 결과 확인

{% highlight python%} 
#10개의 확인 이미지 추출
sample_size = 10
samples = sess.run(decoder, feed_dict={X:mnist.test.images[:sample_size]}) # 디코더로 생성해낸 결과

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2)) # 이미지를 2줄로 보여줄 수 있도록 셋팅

for i in range(sample_size):
    ax[0][i].set_axis_off() # 입력된 이미지
    ax[1][i].set_axis_off() # 생성된 이미지(출력값)
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28))) #imshow : 이미지 출력함수
    ax[1][i].imshow(np.reshape(samples[i], (28,28)))

plt.show()

{% endhighlight %}

![gan1]({{ site.url }}/img/gan1_1.PNG)
