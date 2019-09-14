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
 > [White Whale](https://twinw.tistory.com/247)


앞에서 손실함수에 대해 이해하였으니 이제 이 손실함수의 값을 어떻게 줄여가느냐에 대해 알아봅시다. 손실함수의 값을 어떻게 줄여나가느냐를 정하는 것은 어떤 optimizer를 사용하느냐에 따라 달라집니다. optimizer에 대해 알아보기 전 `Batch` 라는 개념에 대한 이해가 필요합니다. `Batch`는 가중치 등의 매개 변수의 값을 조정하기 위해 사용하는 데이터의 양을 말합니다. 전체 데이터를 가지고 매개 변수의 값을 조정할 수도 있고 정해준 양의 데이터만 가지고도 매개 변수의 값을 조정할 수 있습니다. 

## Batch Gradient Descent(배치 경사 하강법)
이 방법은 가장 기본적인 경사 하강법입니다. 배치경사 하강법은 옵티마이저 중 하나로 loss를 구할때 전체 데이터를 고려합니다. 머신러닝에서는 1번의 훈련횟수를 1 epoch라고 하는데 배치 경사 하강법은 한번의 epoch에 모든 매개변수 업데이트를 단 한번 수행합니다. 이 방법은 전체 데이터를 고려해서 학습하므로 epoch당 시간이 오래 걸리며 메모리를 크게 요구한다는 단점이 있으나 글로벌 미니멈을 찾을 수 있다는 장점이 있습니다. 수식과 케라스 소스 코드는 다음과 같습니다. 

### 수식
* $$ W(t+1) = W(t) - \alpha \frac{\rho}{\rho w} Cost(w) $$

### Keras 
{% highlight python %} 
model.fit(X_train, y_train, batch_size=len(trainX))
{% endhighlight %}

## Stochastic Gradient Descent(SGD)
기존의 배치경사하강법은 전체 데이터에 대해서 계산을 하다보니 시간이 너무 오래 걸린다는 단점이 있습니다. `확률적 경사 하강법`은 매개변수 값을 조정 시 전체 데이터가 아니라 랜덤으로 추출한 한 개의 데이터에 대해서만 가중치를 계산하고 조절하는 방법입니다. 하나의 데이터만을 사용하므로 계산 속도가 더 빨라질 것이라고 추측할 수 있겠죠. 하지만 매개변수의 변경폭이 불안정하고 때로는 배치 경사 하강법보다 정확도가 낮을 수도 있다는 단점을 가지고 있습니다. 

### 수식

수식은 경사하강법과 같습니다.

* $$ W(t+1) = W(t) - \alpha \frac{\rho}{\rho w} Cost(w) $$

단 $$Cost(w)$$에 사용되는 $$x$$, 즉 입력 데이터의 수가 전체가 아닌 확률적으로 선택된 한 개만 사용됩니다. 수식에서 $$\alpha$$ 는 leraning rate를 뜻합니다.

### 코드

#### Python
{% highlight python %} 
weight[i] += - learning_rate * gradient
{% endhighlight %}

#### Keras 
{% highlight python %} 
keras.optimizers.SGD(lr=0.1)
{% endhighlight %}

{% highlight python %} 
model.fit(X_train, y_train, batch_size=1)
{% endhighlight %}

## Mini-Batch Gradient Descent
전체 데이터도 아니고, 1개의 데이터도 아니고 정해진 양에 대해서만 계산하여 매개 변수의 값을 조정하는 경사 하강법을 미니 배치 경사 하강법이라고 합니다. 미니 배치 경사 하강법은 전체 데이터를 계산하는 것보다 빠르며, SGD보다 안정적이라는 장점이 있습니다. SGD라고 한다면 실질적으로는 Mini-Batch 경사하강법을 얘기하기도 합니다. 실제로 가장 많이 사용되는 경사 하강법입니다. 

### 수식

수식은 위의 방법들과 동일합니다. 

* $$ W(t+1) = W(t) - \alpha \frac{\rho}{\rho w} Cost(w) $$

여기서는 $$Cost(w)$$에 사용되는 $$x$$, 입력 데이터의 수가 확률적으로 선택된 부분이 됩니다.

### 코드

#### Keras
{% highlight python %} 
keras.optimizers.SGD(lr=0.1)
{% endhighlight %}

{% highlight python %} 
model.fit(X_train, y_train, batch_size=32)
{% endhighlight %}

## Momentum
모멘텀(Momentum)은 관성, 탄력, 가속도라는 뜻입니다. 단어 그대로 모멘텀은 경사 하강법에 관성을 더해줍니다. 경사 하강법과 마찬가지로 매번 기울기를 구하지만 가중치를 수정하기 전 계산된 접선의 기울기에 한 시점(step) 전의 접선의 기울기값을 일정한 비율만큼 반영합니다. 따라서 양(+) 방향과 음(-) 방향이 순차적으로 일어나는 지그재그 현상이 줄어들고 이전 이동 값을 고려해 일정 비율만큼 다음 값을 결정하게 되므로 관성의 효과를 낼 수 있습니다. 이렇게 하면 마치 언덕에서 공이 내려올 때, 중간에 작은 웅덩이에 빠지더라도 관성의 힘으로 넘어서는 효과를 줄 수 있습니다.

다시 말해 로컬 미니멈에 도달하였을 때, 기울기가 0이라서 기존의 경사 하강법이라면 이를 글로벌 미니멈으로 잘못 인식하여 계산이 끝났을 상황이라도 모멘텀. 즉, 관성의 힘을 빌리면 값이 조절되면서 로컬 미니멈에서 탈출하는 효과를 얻을 수도 있습니다. 케라스에서는 다음과 같이 사용합니다.

### 수식

* $$ V(t) = m * V(t-1) - \alpha \frac{\rho}{\rho w} Cost(w) $$
* $$ W(t+1) = W(t) + V(t) $$

여기서 $$\alpha$$는 leraning rate, $$m$$은 momentum 계수 입니다. 

### 코드
#### Python
{% highlight ruby %} 
v = m * v - learning_rate * gradient
weight[i] += v
{% endhighlight %}
#### Tensorflow
{% highlight ruby %} 
optimize = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(loss)
{% endhighlight %}
#### Keras
{% highlight ruby %} 
keras.optimizers.SGD(lr = 0.01, momentum= 0.9)
{% endhighlight %}


## Adagrad

매개변수들은 각자 의미하는 바가 다른데, 모든 매개변수에 동일한 학습률(learning rate)을 적용하는 것은 비효율적입니다. 따라서 Adagrad는 변수의 업데이트 횟수에 따라 learning ratae를 조절하는 옵션이 추가된 방법입니다. 여기서 변수는 가중치 W 벡터 하나의 값 W[i]를 의미합니다. Adagrad는 각 매개변수에 서로 다른 학습률을 적용시킵니다. 이 때, 변화가 많은 매개변수는 학습률이 작게 설정되고 변화가 적은 매개변수는 학습률을 높게 설정시킵니다. 이는 많이 변화한 변수는 최적값에 근접했을 것이라는 가정하에 작은 크기로 이동하면서 세밀한 값을 조정하고, 반대로 적게 변화한 변수들은 학습률을 크게하여 빠르게 loss값을 줄입니다.

Adagrad는 같은 입력 데이터가 여러번 학습되는 학습모델에 유용하게 쓰이는데 대표적으로 언어와 관련된 word2vec이나 GloVe에 유용합니다. 이는 학습 단어의 등장 확률에 따라 변수의 사용 비율이 확연하게 차이나기 때문에 많이 등장한 단어는 가중치를 적게 수정하고 적게 등장한 단어는 많이 수정할 수 있기 때문입니다.

### 수식
* $$ G(t) = G(t-1) -(\frac{\rho}{\rho w(t)} Cost(w(t)))^2 = \sum^{t}_{i=0} (\frac{\rho}{\rho w(i)} Cost(w(i)))^2 $$
* $$ W(t+1) = W(t)-\alpha * \frac{1}{\sqrt{G(t)+e}} * \frac{\rho}{\rho w(i)} Cost(w(i)) $$

G(t)의 수식을 보면 현재 gradient 제곱에 G(t-1) 값이 더해집니다. 이는 각 step의 모든 gradient에 대한 sum of squares 라는 것을 뜻합니다. W(t+1)을 구하는 식에서 G(t)는 $$\epsilon$$ 값과 더해진 후 루트가 적용되고 $$\alpha$$ 에 나누어 집니다. 여기서 $$\epsilon$$은 아주 작은 상수를 의미하며, 0으로 나누는 것을 방지합니다. 그리고 $$\alpha$$는 learning rate를 나타내며 G(t)의 크기에 따라 값이 변합니다. 

### 코드

#### Python
{% highlight ruby %} 
g += gradient**2
weight[i] += - learning_rate ( gradient / (np.sqrt(g) + e)
{% endhighlight %}

#### Tensorflow
{% highlight ruby %} 
optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
{% endhighlight %}

#### Keras
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
