---
layout: post
title: Over fitting
excerpt: "Over fitting의 개념과 예방"
categories: [deep learning]
comments: true
---

주로 참고한 블로그 글은 다음과 같습니다.
{: .notice}
 
 > [UMBUM](https://umbum.tistory.com/222)

## Over fitting 이란
한국어로 과적합. 단어에서 알수 있듯이 학습하는 데이터 (training data)에 대해 너무나 잘 학습하는 것을 의미합니다. 그런데 너무나 잘 학습하는게 왜 문제일까요? 

![over_1]({{ site.url }}/img/overfitting.png)

그림에서 처럼 왼쪽은 제대로 학습이 되지 않은 상태, 가운데는 최적의 학습상태, 그리고 오른쪽이 overfitting 상태입니다. 만약 오른쪽 그림처럼 overfitting 인 상태라면 학습 데이터에 대해서만 정확하게 학습되어 실제 데이터에서는 좋은 성능을 보여주지 못하게 됩니다. 따라서, 아래 그림에서 처럼 traing data와 test data에 대해서 error 값이나 accuracy 값이 과하게 차이가 난다면 overfitting이 일어났다고 할 수 있습니다. 
한마디로 ovefitting 이란 `학습 데이터에` `대해 과하게` `학습하여` `실제 데이터에` `대한 오차가` `증가하는 현상` 입니다.

![over_2]({{ site.url }}/img/overfitting2.png)

## Bias 와 Variance

over fitting을 좀더 자세히 알기 위해서는 bias와 varience에 대한 개념을 알아야 합니다.

![over_2]({{ site.url }}/img/bias.PNG)
* Bias: 실제 값에서 멀어진 척도
* variance: 예측된 값들이 서로 얼마나 떨어져 있는가 척도

bias와 variance를 줄이는 것이 딥러닝의 목표라 할 수 있습니다. 이때 `bias`가 높아진다는 것은 `underfitting`이 일어나는 것이고 `variance`가 높아진다는 것은 `overfitting`이 일어나는 것이라 할 수 있습니다. 또한 `bias와 variace`는 밑에 그림에서 볼 수 있듯 `trade off` 관계를 가지고 있습니다.
![over_2]({{ site.url }}/img/tradeoff.png)
정리하자면
* 높은 bias = 낮은 variance = under fitting
* 높은 variance = 낮은 bias = over fitting

따라서 좋은 성능을 지닌 딥러닝 모델을 얻기 위해서는 이 두 지점 사이의 적당한 균형점에 도달 할 필요가 있습니다. 보통 underfitting에 대한 문제는 쉽게 해결할 수 있습니다. 모델을 복잡하게 만들고 데이터의 parameter 개수를 늘리죠. 가장 큰 문제는 overfitting 입니다. 그렇다면 어떻게 over fitting을 방지할 수 있을까요?

## Over fitting 예방책
### validation data
### dropout
드롭아웃은 각 계층마다 일정 비율의 뉴런을 임의로 정해 drop 시켜 나머지 뉴런들만 학습하도록 하는 방법입니다. 네트워크 내부에서 이루어지는 `ensemble learning` 이라고 생각해도 좋습니다. 
> `ensenmble learning`은 개별적으로 학습시킨 여러 모델의 출력을 종합해 추론하는 방식입니다. 뉴럴 네트워크를 개별적으로 학습시키고, 각 네트워크에 같은 input을 주어 나온 출력 뉴런 각각의 평균을 구한 다음 여기서 가장 큰 값을 정답으로 판정하면 되기 때문에 간단히 구현할 수 있습니다. 
> * $$1/N \sum_{i}^{N} output_i$$

![over_2]({{ site.url }}/img/dropout.png)

### L2 regularization
가장 일반적으로 사용하는 regularization 기법으로 가중치가 클 수록 큰 패널티를 부과하여 오버피팅을 억제하는 방법입니다. 패널티를 부과하는 방법은 loss function에 $$ 1/2 \lambda \sum W^2 $$ 을 더해줍니다. 이 값을 미분한 값은 $$ \lambda W$$이고 오차역전파를 통해 계산한 기울기에 $$\lambda W$$를 더하게 되어 가중치 값이 그만큼 보정됩니다. 
* $$ L = 1/2m \sum_{i=1}^{m} (h_\theta(x^{(i)})-y^{(i)})^2 + 1/2 \lambda \sum W^2 $$ 

참고로 L1 regularization은 loss function에 $$ \lambda W$$ 을 더해주는 방법입니다. 일반적으로는 L2 regularization을 사용하는 방법이 좋은 결과를 얻을 수 있다고 알려져 있습니다.

### Early stopping
보통 epoch 수 만큼 반복해서 학습합니다. 그러나 계속 반복학습하게 되면 어느시점에서 overtrainig이 일어나게 됩니다. validation accuracy가 더이상 올라가지 않을 때 학습을 멈추는 것을 early stopping이라고 합니다. 

#### Keras
Keras에서는 라이브러리를 이용하여 쉽게 구현 할 수 있도록 해놓았습니다.
`EarlyStopping` 콜백을 사용하면 정해진 epoch 동안 모니터링 지표가 향상되지 않을 때 훈련을 중지할 수 있습니다. 일반적으로 이 콜백은 훈련하는 동안 모델을 계속 저장해주는 `ModelCheckpoint`와 함께 사용합니다. (가장 좋은 모델만 저장 가능)
{% highlight ruby %} 
    #1 epoch 동안 valldation accuracy가 더 향상되지 않으면 중단
    EarlyStopping(monitor='val_acc', patience=1) 
    # set checkpointer and save model
    checkpointer = ModelCheckpoint(filepath=save_path+'model.hdf5', verbose=1, save_best_only=True)
    
{% endhighlight %}

