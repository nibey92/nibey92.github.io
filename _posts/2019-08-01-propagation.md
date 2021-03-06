---
layout: post
title: Propagation
excerpt: "딥러닝의 학습과정을 살펴보자"
categories: [deep learning]
comments: true
---

딥러닝에서 역전파를 이해하는 것은 아주 중요합니다. 이번 시간에는 간단한 모델을 가지고 딥러닝이 학습하는 과정과 역전파를 계산하는 과정을 살펴보도록 하겠습니다. 또한 파이썬으로 간단히 구현도 함께 해보도록 하겠습니다. 시작하기 앞서 이번 글은 [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/37406) 의 ``역전파 이해하기`` 글을 공부하고 그 내용을 제 나름대로 풀어 쓴 것임을 밝힙니다.
 
 > [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/37406)

## Neural Network Overview
예제를 위해 사용될 인공 신경망은 앞에서 소개한 사이트의 모델과 같은 것을 사용합니다. 이 인공 신경망은 입력층, 은닉층, 출력층 3개의 층을 가집니다. 또한 각 층에는 두개의 뉴런이 있습니다. 은닉층과 출력층의 모든 뉴런은 시그모이드 함수를 활성화 함수로 사용합니다.
![propa]({{ site.url }}/img/propa.PNG)

#### 은닉층
* W : 입력층에서 은닉층 방향으로 향하는 가중치 입니다.
* z : 입력층의 x 값에 가중치 W가 곱해진 값입니다.
* h : z 값이 시그모이드 함수를 지난 후 값으로 은닉층의 최종 출력값입니다.

#### 출력층
* U : 은닉층에서 출력층 방향으로 향하는 가중치 입니다.
* t : 은닉층의 출력값 h에 가중치 U가 곱해진 값입니다.
* o : t 값이 시그모이드 함수를 지난 후의 값으로 출력층의 최종 출력값입니다.
 
이번 역전파 예제에서는 인공 신경망에 존재하는 모든 가중치 W와 U에 대해서 역전파를 통해 업데이트하는 것을 목표로 합니다. 해당인공 신경망은 편향 b는 고려하지 않습니다.

## Forward Propagation
![propa2]({{ site.url }}/img/propa2.PNG)
역전파를 계산하기 전에 먼저 순전파를 진행하여 딥러닝이 학습하는 순서를 먼저 알아봅시다. x, W, U, 그리고 실제값 y는 모두 상수로 주어집니다. 그림에서 모든 값들을 표시하였습니다. 이 값들을 코드에 먼저 초기값으로 적어놓습니다.

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
* $$ z_1 = W_{11}*x_1 + W_{21}*x_2 = 0.08 $$
* $$ z_2 = W_{12}*x_1 + W_{22}*x_2 = 0.11 $$
* $$ h_1 = sigmoid(z_1) = 0.51998934 $$
* $$ h_2 = sigmoid(z_2) = 0.0.52747230 $$

#### 출력층
* $$ t_1 = U_{11}*h_1 + U_{21}*t_2 = 0.44498412 $$
* $$ t_2 = U_{12}*h_1 + U_{22}*t_2 = 0.68047592 $$
* $$ o_1 = sigmoid(t_1) = 0.60944600 $$
* $$ o_2 = sigmoid(t_2) = 0.66384491 $$

#### 오차
이제 해야할 일은 예측값과 실제값의 오차를 계산하기 위한 오차함수를 선택하는 것입니다. 오차를 계산하기 위한 손실함수는 평균제곱오차(Mean squared error, MSE)를 사용합니다. 각 오차를 모두 더하면 전체 오차가 됩니다. 
> $$ MSE = 1/N \sum (y_i-\hat{y})^2 $$

* $$ E_1 = \frac{1}{2}(y_1-o_1)^2 = 0.02193381 $$
* $$ E_2 = \frac{1}{2}(y_2-o_2)^2 = 0.00203809 $$
* $$ E_{tot} = E_1 + E_2 = 0.02397190 $$

코드로 구현하여 계산을 해보겠습니다. 순전파의 경우 간단한 곱셈과 시그모이드 함수로 값을 얻을 수 있기 때문에 쉽게 구현이 가능합니다.

{% highlight ruby %} 
def sigmoid(x):
  return 1/(1+np.exp(-x))

z1 = x1*W11+x2*W21
z2 = x1*W12+x2*W22
h1 = sigmoid(z1)
h2 = sigmoid(z2)

t1 = h1*U11+h2*U21
t2 = h1*U12+h2*U22
o1 = sigmoid(t1)
o2 = sigmoid(t2)

E1=0.5*(y1-o1)**2
E2=0.5*(y2-o2)**2
E=E1+E2

{% endhighlight %}

코드를 실행시켜 결과를 비교해 봅니다. 
{% highlight ruby %} 

------hidden layer------
z1: 0.08
z2: 0.11
h1: 0.519989340156
h2: 0.527472304345
------ out layer--------
t1: 0.444984124808
t2: 0.680475920716
o1: 0.609446004997
o2: 0.663844909698
------ Entrophy---------
E1: 0.0219338145045
E2: 0.00203808624719
E: 0.0239719007517
------------------------

{% endhighlight %}

이렇게 해서 오늘은 간단한 모델을 가지고 순전파가 진행하는 과정을 알아보았습니다. 이제 이 결과값 (여기서는 MSE를 사용해서 얻은 error값)을 가지고 딥러닝이 어떻게 가중치를 업데이트하는지 알아야 합니다. 여기에 사용되는 것이 바로 역전파(back propagation) 입니다. 역전파의 계산 과정은 순전파 보다 훨신 복잡합니다. 역전파에 대해서는 다음 포스팅에서 쓰도록 하겠습니다.
