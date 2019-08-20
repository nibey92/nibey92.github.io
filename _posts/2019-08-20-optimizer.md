---
layout: post
title: Optimizer
excerpt: "옵티마이저 개념과 종류를 알아보자"
categories: [deep learning]
comments: true
---

주로 참고한 블로그 글은 다음과 같습니다.
{: .notice}
 
 > [UMBUM](https://umbum.tistory.com/222)


앞에서 손실함수에 대해 이해하였으니 이제 이 손실함수의 값을 어떻게 줄여가느냐에 대해 알아봅시다. 손실함수의 값을 어떻게 줄여나가느냐를 정하는 것은 어떤 optimizer를 사용하느냐에 따라 달라집니다. optimizer에 대해 알아보기 전 `Batch` 라는 개념에 대한 이해가 필요합니다. `Batch`는 가중치 등의 매개 변수의 값을 조정하기 위해 사용하는 데이터의 양을 말합니다. 전체 데이터를 가지고 매개 변수의 값을 조정할 수도 있고 정해준 양의 데이터만 가지고도 매개 변수의 값을 조정할 수 있습니다. 

### Batch Gradient Descent(배치 경사 하강법)
이 방법은 가장 기본적인 경사 하강법입니다. 배치경사 하강법은 옵티마이저 중 하나로 loss를 구할때 전체 데이터를 고려합니다. 머신러닝에서는 1번의 훈련횟수를 1 epoch라고 하는데 배치 경사 하강법은 한번의 epoch에 모든 매개변수 업데이트를 단 한번 수행합니다. 이 방법은 전체 데이터를 고려해서 학습하므로 epoch당 시간이 오래 걸리며 메모리를 크게 요구한다는 단점이 있으나 글로벌 미니멈을 찾을 수 있다는 장점이 있습니다. 
{% highlight ruby %} 
model.fit(X_train, y_train, batch_size=len(trainX))
{% endhighlight %}

### Stochastic Gradient Descent(SGD)
기존의 배치경사하강법은 전체 데이터에 대해서 계산을 하다보니 시간이 너무 오래 걸린다는 단점이 있습니다. `확률적 경사 하강법`은 매개변수 값을 조정 시 전체 데이터가 아니라 랜덤으로 선택한 하나의 데이터에 대해서만 계산하는 방법입니다. 더 적은 데이터를 사용하므로 더 빠르게 계산할 수 있습니다. 매개변수의 변경폭이 불안정하고 때로는 배치 경사 하강법보다 정확도가 낮을 수도 있지만 속도만큼은 배치 경사 하강법보다 빠르다는 장점이 있습니다. 케라스에는 아래와 같이 사용합니다.
{% highlight ruby %} 
model.fit(X_train, y_train, batch_size=1)
{% endhighlight %}

### Mini-Batch Gradient Descent
전체 데이터도 아니고, 1개의 데이터도 아니고 정해진 양에 대해서만 계산하여 매개 변수의 값을 조정하는 경사 하강법을 미니 배치 경사 하강법이라고 합니다. 미니 배치 경사 하강법은 전체 데이터를 계산하는 것보다 빠르며, SGD보다 안정적이라는 장점이 있습니다. 실제로 가장 많이 사용되는 경사 하강법입니다.
{% highlight ruby %} 
model.fit(X_train, y_train, batch_size=32)
{% endhighlight %}

### Momentum
모멘텀(Momentum)은 관성이라는 물리학의 법칙을 응용한 방법입니다. 모멘텀 SGD는 경사 하강법에 관성을 더 해줍니다. 모멘텀은 SGD에서 계산된 접선의 기울기에 한 시점(step) 전의 접선의 기울기값을 일정한 비율만큼 반영합니다. 이렇게 하면 마치 언덕에서 공이 내려올 때, 중간에 작은 웅덩이에 빠지더라도 관성의 힘으로 넘어서는 효과를 줄 수 있습니다.

다시 말해 로컬 미니멈에 도달하였을 때, 기울기가 0이라서 기존의 경사 하강법이라면 이를 글로벌 미니멈으로 잘못 인식하여 계산이 끝났을 상황이라도 모멘텀. 즉, 관성의 힘을 빌리면 값이 조절되면서 로컬 미니멈에서 탈출하는 효과를 얻을 수도 있습니다. 케라스에서는 다음과 같이 사용합니다.
{% highlight ruby %} 
keras.optimizers.SGD(lr = 0.01, momentum= 0.9)
{% endhighlight %}

### Adagrad

매개변수들은 각자 의미하는 바가 다른데, 모든 매개변수에 동일한 학습률(learning rate)을 적용하는 것은 비효율적입니다. 아다그라드는 각 매개변수에 서로 다른 학습률을 적용시킵니다. 이 때, 변화가 많은 매개변수는 학습률이 작게 설정되고 변화가 적은 매개변수는 학습률을 높게 설정시킵니다. 케라스에서는 다음과 같이 사용합니다.
{% highlight ruby %} 
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
{% endhighlight %}

### RMSprop

아다그라드는 학습을 계속 진행한 경우에는, 나중에 가서는 학습률이 지나치게 떨어진다는 단점이 있는데 이를 다른 수식으로 대체하여 이러한 단점을 개선하였습니다. 케라스에서는 다음과 같이 사용합니다
{% highlight ruby %} 
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
{% endhighlight %}

### Adam
현재 가장 일반적으로 사용되는 알고리즘이라 할 수 있습니다. 아담은 알엠에스프롭과 모멘텀 두 가지를 합친 듯한 방법으로, 방향과 학습률 두 가지를 모두 잡기 위한 방법입니다. 케라스에서는 다음과 같이 사용합니다.
{% highlight ruby %} 
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
{% endhighlight %}
