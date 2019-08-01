---
layout: post
title: Back Propagation
excerpt: "역전파 이해하기"
categories: [deep learning]
comments: true
---

주로 참고한 블로그 글은 다음과 같습니다.
{: .notice}
 
 > [UMBUM](https://umbum.tistory.com/222)

## Neural Network Overview
예제를 위해 사용될 인공 신경망은 앞에서 소개한 사이트의 모델과 같은 것을 사용합니다. 이 인공 신경망은 입력층, 은닉층, 출력층 3개의 층을 가집니다. 또한 각 층에는 두개의 뉴런이 있습니다. 은닉층과 출력층의 모든 뉴런은 시그모이드 함수를 활성화 함수로 사용합니다.
![propa]({{ site.url }}/img/propa.PNG)

#### 은닉층
* W : 입력층에서 은닉층 방향으로 향하는 가중치 입니다.
* z : 입력층의 x 값에 가중치 W가 곱해진 값입니다.
* h : z 값이 시그모이드 함수를 지난 후 값으로 은닉층의 출력값입니다.

#### 출력층
* U : 은닉층에서 출력층 방향으로 향하는 가중치 입니다.
* t : 은닉층의 출력값 h에 가중치 U가 곱해진 값입니다.
* o : t 값이 시그모이드 함수를 지난 후의 값으로 출력층의 출력값입니다.
 
이번 역전파 예제에서는 인공 신경망에 존재하는 모든 가중치 W와 U에 대해서 역전파를 통해 업데이트하는 것을 목표로 합니다. 해당인공 신경망은 편향 b는 고려하지 않습니다.

## Forward Propagation

역전파를 계산하기 전에 먼저 순전파를 진행하여 딥러닝이 학습하는 순서를 먼저 알아봅시다. x, W, U, 실제값 y는 모두 상수입니다. 그림에서 모든 값들을 표시하였습니다. 이 값들을 코드에 먼저 초기값으로 적어놓습니다.
{% highlight ruby %} 
import numpy as np

x1= 0.1 
x2= 0.2

W11 = 0.3 
W21 = 0.25
W12 = 0.4 
W22 = 0.35

U11 = 0.45
U21 = 0.4
U12 = 0.7
U22 = 0.6

y1 = 0.4
y2 = 0.6

{% endhighlight %}

이제 차례대로 각 층에서의 값들을 계산해봅니다. 
#### 은닉층
$$ z_1 = W_11*x_1 + W_21*x_2 = 0.08 $$
$$ z_2 = W_12*x_1 + W_22*x_2 = 0.11 $$
$$ h_1 = sigmoid(z_1) = 0.52 $$
$$ h_2 = sigmoid(z_2) = 0.53 $$

코드로 간단히 구현하고 계산해봅니다.
{% highlight ruby %} 
z1 = x1*W11+x2*w21
z2 = x1*W12+x2*W22
h1 = sigmoid(z1)
h2 = sigmoid(z2)

print "------hidden layer--------"
print "z1:",z1
print "z2:",z2
print "h1:",h1
print "h2:",h2

{% endhighlight %}

코드 출력 결과는 다음과 같습니다.
{% highlight ruby %} 
------hidden layer--------
z1: 0.08
z2: 0.11
h1: 0.519989340156
h2: 0.527472304345
{% endhighlight %}
