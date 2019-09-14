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

## 1. Batch Gradient Descent(배치 경사 하강법)
이 방법은 가장 기본적인 경사 하강법입니다. 배치경사 하강법은 옵티마이저 중 하나로 loss를 구할때 전체 데이터를 고려합니다. 머신러닝에서는 1번의 훈련횟수를 1 epoch라고 하는데 배치 경사 하강법은 한번의 epoch에 모든 매개변수 업데이트를 단 한번 수행합니다. 이 방법은 전체 데이터를 고려해서 학습하므로 epoch당 시간이 오래 걸리며 메모리를 크게 요구한다는 단점이 있으나 글로벌 미니멈을 찾을 수 있다는 장점이 있습니다. 수식과 케라스 소스 코드는 다음과 같습니다. 

### 수식
* $$ W(t+1) = W(t) - \alpha \frac{\partial}{\partial w} Cost(w) $$

### Keras 
{% highlight python %} 
model.fit(X_train, y_train, batch_size=len(trainX))
{% endhighlight %}

## 2. Stochastic Gradient Descent(SGD)
기존의 배치경사하강법은 전체 데이터에 대해서 계산을 하다보니 시간이 너무 오래 걸린다는 단점이 있습니다. `확률적 경사 하강법`은 매개변수 값을 조정 시 전체 데이터가 아니라 랜덤으로 추출한 한 개의 데이터에 대해서만 가중치를 계산하고 조절하는 방법입니다. 하나의 데이터만을 사용하므로 계산 속도가 더 빨라질 것이라고 추측할 수 있겠죠. 하지만 매개변수의 변경폭이 불안정하고 때로는 배치 경사 하강법보다 정확도가 낮을 수도 있다는 단점을 가지고 있습니다. 

![optimizer_n-1]({{ site.url }}/img/optimizer_n-1.PNG)

> 배치 경사 하강법과 확률적 경사 하강법의 비교 

### 수식

수식은 경사하강법과 같습니다.

* $$ W(t+1) = W(t) - \alpha \frac{\partial}{\partial w} Cost(w) $$

단 $$Cost(w)$$에 사용되는 $$x$$, 즉 입력 데이터의 수가 전체가 아닌 확률적으로 선택된 한 개만 사용됩니다. 수식에서 $$\alpha$$ 는 leraning rate를 뜻합니다.

### Python
{% highlight python %} 
weight[i] += - learning_rate * gradient
{% endhighlight %}

### Keras 
{% highlight python %} 
keras.optimizers.SGD(lr=0.1)
{% endhighlight %}

{% highlight python %} 
model.fit(X_train, y_train, batch_size=1)
{% endhighlight %}

## 3. Mini-Batch Gradient Descent
전체 데이터도 아니고, 1개의 데이터도 아니고 정해진 양에 대해서만 계산하여 매개 변수의 값을 조정하는 경사 하강법을 미니 배치 경사 하강법이라고 합니다. 미니 배치 경사 하강법은 전체 데이터를 계산하는 것보다 빠르며, SGD보다 안정적이라는 장점이 있습니다. SGD라고 한다면 실질적으로는 Mini-Batch 경사하강법을 얘기하기도 합니다. 실제로 가장 많이 사용되는 경사 하강법입니다. 

### 수식

수식은 위의 방법들과 동일합니다. 

* $$ W(t+1) = W(t) - \alpha \frac{\partial}{\partial w} Cost(w) $$

여기서는 $$Cost(w)$$에 사용되는 $$x$$, 입력 데이터의 수가 확률적으로 선택된 부분이 됩니다.

### Keras
{% highlight python %} 
keras.optimizers.SGD(lr=0.1)
{% endhighlight %}

{% highlight python %} 
model.fit(X_train, y_train, batch_size=32)
{% endhighlight %}

## 4. Momentum
모멘텀(Momentum)은 관성, 탄력, 가속도라는 뜻입니다. 단어 그대로 모멘텀은 경사 하강법에 관성을 더해줍니다. 경사 하강법과 마찬가지로 매번 기울기를 구하지만 가중치를 수정하기 전 계산된 접선의 기울기에 한 시점(step) 전의 접선의 기울기값을 일정한 비율만큼 반영합니다. 따라서 양(+) 방향과 음(-) 방향이 순차적으로 일어나는 지그재그 현상이 줄어들고 이전 이동 값을 고려해 일정 비율만큼 다음 값을 결정하게 되므로 관성의 효과를 낼 수 있습니다. 이렇게 하면 마치 언덕에서 공이 내려올 때, 중간에 작은 웅덩이에 빠지더라도 관성의 힘으로 넘어서는 효과를 줄 수 있습니다.

다시 말해 로컬 미니멈에 도달하였을 때, 기울기가 0이라서 기존의 경사 하강법이라면 이를 글로벌 미니멈으로 잘못 인식하여 계산이 끝났을 상황이라도 모멘텀. 즉, 관성의 힘을 빌리면 값이 조절되면서 로컬 미니멈에서 탈출하는 효과를 얻을 수도 있습니다. 케라스에서는 다음과 같이 사용합니다. 

### 수식

* $$ V(t) = m * V(t-1) - \alpha \frac{\partial}{\partial w} Cost(w) $$
* $$ W(t+1) = W(t) + V(t) $$

여기서 $$\alpha$$는 leraning rate, $$m$$은 momentum 계수 입니다. 보통 0.9로 설정하며 교차 검증을 한다면 0.5에서 시작하여 0.9, 0.95, 0.99 순서로 증가시켜 검증합니다. 예시로 맨 처음 gradient($$\alpha*\frac{\partial}{\partial w}*Cost(w) $$)의 값이 0.5이고 두 번째 gradient 값이 -0.3이라 할 때 $$m$$이 0.9라면 $$V(1)$$은 -0.5, $$V(2)$$는 0.9 * -0.5 +0.3 = -0.45 + 0.3 = -0.15가 됩니다. 이처럼 gradient의 방향이 변경되어도 이전 방향과 크기에 영향받아 다른 방향으로 가중치가 변경될 수 있습니다.

### Python
{% highlight python %} 
v = m * v - learning_rate * gradient
weight[i] += v
{% endhighlight %}
### Tensorflow
{% highlight python %} 
optimize = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9).minimize(loss)
{% endhighlight %}
### Keras
{% highlight python %} 
keras.optimizers.SGD(lr = 0.01, momentum= 0.9)
{% endhighlight %}

## 5. Nesterov Accelrated Gradient, NAG

NAG는 momentum 값과 gradient 값이 더해저 실제(actual) 값을 만드는 기존 momentum과는 달리 momentum 값이 적용된 지점에서 gradient 값이 계산됩니다. 

### 수식

![optimizer_n-3]({{ site.url }}/img/optimizer_n-3.PNG)

수식을 보면 gradient를 구할 때 분모($$\partial w$$)의 가중치 (W)에 먼저 $$mV(t-1)$$ 값을 더해 계산하는 것입니다. 이 단계를 추가함으로써 $$V(t)$$를 계산하기 전 momentum 방법으로 인해 이동될 방향을 미리 예측하고 해당 방향으로 얼마간 미리 이동한 뒤 gradient를 계산하는 효과를 얻을 수 있습니다. 즉 한단계를 미리 예측하여 불필요한 이동을 줄입. 

### Python
{% highlight python %} v = m * v - learning_rate * gradient(weight[i-1]+m*v)
weight[i] += v
{% endhighlight %}
### Tensorflow
{% highlight python %} 
optimize = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9,use_nesterov=True).minimize(loss)
{% endhighlight %}
### Keras
{% highlight python %} 
keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
{% endhighlight %}

## 6. Adagrad

매개변수들은 각자 의미하는 바가 다른데, 모든 매개변수에 동일한 학습률(learning rate)을 적용하는 것은 비효율적입니다. 따라서 Adagrad는 변수의 업데이트 횟수에 따라 learning ratae를 조절하는 옵션이 추가된 방법입니다. 여기서 변수는 가중치 $$W$$ 벡터 하나의 값 $$W[i]$$를 의미합니다. Adagrad는 각 매개변수에 서로 다른 학습률을 적용시킵니다. 이 때, 변화가 많은 매개변수는 학습률이 작게 설정되고 변화가 적은 매개변수는 학습률을 높게 설정시킵니다. 이는 많이 변화한 변수는 최적값에 근접했을 것이라는 가정하에 작은 크기로 이동하면서 세밀한 값을 조정하고, 반대로 적게 변화한 변수들은 학습률을 크게하여 빠르게 loss값을 줄입니다.

Adagrad는 같은 입력 데이터가 여러번 학습되는 학습모델에 유용하게 쓰이는데 대표적으로 언어와 관련된 word2vec이나 GloVe에 유용합니다. 이는 학습 단어의 등장 확률에 따라 변수의 사용 비율이 확연하게 차이나기 때문에 많이 등장한 단어는 가중치를 적게 수정하고 적게 등장한 단어는 많이 수정할 수 있기 때문입니다.

### 수식

![optimizer_n-4]({{ site.url }}/img/optimizer_n-4.PNG)

$$G(t)$$의 수식을 보면 현재 gradient 제곱에 $$G(t-1)$$ 값이 더해집니다. 이는 각 step의 모든 gradient에 대한 sum of squares 라는 것을 뜻합니다. $$W(t+1)$$을 구하는 식에서 $$G(t)$$는 $$\epsilon$$ 값과 더해진 후 루트가 적용되고 $$\alpha$$ 에 나누어 집니다. 여기서 $$\epsilon$$은 아주 작은 상수를 의미하며, 0으로 나누는 것을 방지합니다. 그리고 $$\alpha$$는 learning rate를 나타내며 $$G(t)$$의 크기에 따라 값이 변합니다. 

### Python
{% highlight python %} 
g += gradient**2
weight[i] += - learning_rate ( gradient / (np.sqrt(g) + e)
{% endhighlight %}
### Tensorflow
{% highlight python %} 
optimizer = tf.train.AdagradOptimizer(learning_rate=0.01).minimize(loss)
{% endhighlight %}
### Keras
{% highlight python %} 
keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)
{% endhighlight %}

## 7. RMSprop

Adagrad로 학습을 계속 진행하는 경우에 나중에 가서는 학습률이 지나치게 떨어진다는 단점이 있습니다. 그 이유는 Adagrad의 G(t) 값이 무한히 커지는 경우가 있기 때문입니다. RMSprop는 이를 방지하고자 지수 이동평균을 이용한 방법입니다. 지수 이동평균 이란 쉽게 말해 최근 값을 더 잘 반영하기 위해 최근 값과 이전 값에 각각 가중치를 주어 계산하는 방법입니다. 

### 수식
먼저 지수 이동평균의 수식을 알아보겠습니다. 

* $$ x_k = \alpha p_k + (1-\alpha)x_{k-1} $$
* $$ where \; \alpha = \frac{2}{N+1} $$

위 식에서 지수 이동평균값은 $$x$$, 현재 값은 $$p$$, 가중치는 $$\alpha$$이며 아래 첨자 $$k$$는 step 혹은 time, 마지막으로 $$N$$은 값의 개수라고 보시면 됩니다. 만약 처음부터 현재까지 계산을 하게 된다면 $$N$$과 $$k$$ 값은 같으며 가중치 $$\alpha$$ 는 $$N$$이 작을 수록 커집니다. 계산 식을 풀어 써보면 아래와 같습니다. 

![optimizer_n-2]({{ site.url }}/img/optimizer_n-2.PNG)

### Python
{% highlight python %} 
g = gamma * g + (1 - gamma) * gradient**2
weight[i] += -learning_rate * gradient / (np.sqrt(g) + e)
{% endhighlight %}
### Tensorflow
{% highlight python %} 
optimize = tf.train.RMSPropOptimizer(learning_rate=0.01,decay=0.9,momentum=0.0,epsilon=1e-10).minimize(cost)
{% endhighlight %}
### Keras
{% highlight python %} 
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
{% endhighlight %}

## 8. Adam (Adaptive Moment Estimation)
현재 가장 일반적으로 사용되는 알고리즘이라 할 수 있습니다. Adam은 RMSprop와 Momentum 두가지를 합친 듯한 방법으로, 방향과 학습률 두 가지를 모두 잡기 위한 방법입니다. RMSprop의 특징인 gradient의 제곱을 지수평균한 값을 사용하며 Momentum의 특징으로 gradient를 제곱하지 않은 값을 사용하여 지수평균을 구하고 수식에 활용합니다. 

### 수식

![optimizer_n-5]({{ site.url }}/img/optimizer_n-5.PNG)

기존 RMSprop와 momentum과 다르게 $$M(t)$$와 $$V(t)$$가 바로 $$W(t+1)$$ 수식에 들어가는 것이 아니라 $$M(t)$$와 $$V(t)$$가 들어갑니다. 이 부분을 논문에서는 bias가 수정된 값으로 변경하는 과정이라고 합니다. 이전에 저희가 알아야 할 것은 초기 $$M(0)$$와 $$V(0)$$ 값이 0으로 초기화 되는데 시작값이 0이기 때문에 이동 평균을 구하면 0으로 편향된 값 추정이 발생할 수 있습니다. 특히 초기 감쇠 속도가 작은 경우 (즉, $$\beta$$가 1에 가까울 때 )에 발생합니다. 이를 방지하기 위해 $$1-\beta^t$$ 값을 나누어 bias 보정을 해줍니다. $$1-\beta^t$$는 $$M(t)$$와 $$V(t)$$의 기대값을 구하는 과정에서 찾을 수 있습니다. 추가적으로 $$\alpha=0.001$$, $$\beta_1 = 0.9$$. $$\beta_2 = 0.999$$, $$\epsilon = 10^{-8}$$ 이 가장 좋은 default 값이라 논문에 명시되어 있다고 합니다. 

### Tensorflow
{% highlight python %} 
optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08 ).minimize(loss)
{% endhighlight %}
### Keras
{% highlight python %} 
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
{% endhighlight %}

## 9. AdaDelta(Adaptive Delta)
AdaDelta는 Adagrad, RMSprop, Momentum 모두를 합친 경사하강법입니다. 크게 두가지 특징이 있습니다. ``첫번째``는 Adagrad 특징인 모든 step의 gradient 제곱의 합을 window size를 두어 window size 만큼의 합으로 변경합니다. 이후 RMSprop와 똑같이 지수 이동평균을 적용합니다. ``두번째``는 Hessian 근사법을 이용한 단위(Units) 수정입니다. 

### 수식

![optimizer_n-7]({{ site.url }}/img/optimizer_n-7.PNG)

논문에서는 ``가중치와 가중치 변화량의 단위가 같아야 한다``라고 명시되어 있습니다. 그리고 SGD, Momentum Adagrad는 업데이트가 기울기 양의 비율을 포함하므로 정확한 단위를 가지지 않고 따라서 업데이트는 단위가 없다라고 설명합니다. 그리고 위 수식을 보면 알 수 있듯이 $$\bigtriangleup x$$의 단위는 $$x$$의 단위가 아닌 $$x$$ 단위의 역수와 관계가 있다는 것을 알 수 있습니다. 반대로 AdaDelta의 경우 Newton's method를 이용하여 아래 수식과 같이 $$\bigtriangleup x$$와 $$x$$의 단위간의 관계를 만듭니다. 

* $$\bigtriangleup x  \propto H^{-1} g \propto \frac{\frac{\partial f}{\partial z}}{\frac{\partial^2 f}{\partial x^2}} \propto units \; of \; x $$

특정 함수에 대해 gradient는 일차미분(first derivative)를 나타내는 반면 Hessian은 함수의 이차미분(second derivative)를 나타냅니다. 즉, Hessian은 함수의 곡률(curvature) 특성을 나타내는 행렬로서 최적화 문제에 적용할 경우 Hessian을 이용하면 특정 지점 근처에서 함수를 2차 항까지 근사시킬 수 있습니다. (second-order Taylor expansion)

더 큰 장점은 critical point의 종류를 판별할 수 있다는 것입니다. gradient는 함수에서 일차미분이 0이 되는 부분인 critical point (stationary point)를 찾을 수 있지만 Hessian은 해당 지점이 극대인지 극소인지 아니면 Saddle point인지 알 수 있습니다.

경사 하강법에서는 First Order Methods가 적용된 하강법에서는 1차 미분만 하여 gradient가 0인 지점을 찾습니다. 그렇다보니 saddle point에서 문제가 발생할 수 있습니다. 반면 Second Order Methods를 사용한 AdaDelta는 saddle point에서 문제가 발생하지 않지만 2차 미분까지 하기때문에 계산속도가 상대적으로 느립니다.

![optimizer_n-6]({{ site.url }}/img/optimizer_n-6.PNG)

$$G(t) \rightarrow \bigtriangleup w(t) \rightarrow S(t) \rightarrow W(t+1) $$ 순서로 계산이 되며 loop 크기만큼 반복됩니다. 

### Tensorflow
{% highlight python %} 
optimizer = tf.train.adadeltaoptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08).minimize(loss)
{% endhighlight %}
### Keras
{% highlight python %} 
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
{% endhighlight %}
