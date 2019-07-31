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

![over_2]({{ site.url }}/img/overfitting2.png)

`overfitting이란 학습데이터에 대해 과하게 학습하여 실제 데이터에 대한 오차가 증가하는 현상` 이라 할 수 있습니다.

## Bias 와 Variance

over fitting을 좀더 자세히 알기 위해서는 bias와 varience에 대한 개념을 알아야 합니다.

![over_2]({{ site.url }}/img/bias.PNG)

* Bias: 실제 값에서 멀어진 척도

* variance: 예측된 값들이 서로 얼마나 떨어져 있는가 척도
![over_2]({{ site.url }}/img/tradeoff.png)

## Over fitting 예방책
### validation data
### dropout
드롭아웃은 각 계층마다 일정 비율의 뉴런을 임의로 정해 drop 시켜 나머지 뉴런들만 학습하도록 하는 방법입니다. 네트워크 내부에서 이루어지는 ensemble learning 이라고 생각해도 좋습니다. 
> ensenmble learning은 개별적으로 학습시킨 여러 모델의 출력을 종합해 추론하는 방식입니다. 뉴럴 네트워크를 개별적으로 학습시키고, 각 네트워크에 같은 input을 주어 나온 출력 뉴런 각각의 평균을 구한 다음 여기서 가장 큰 값을 정답으로 판정하면 되기 때문에 간단히 구현할 수 있습니다. 
> $$1/N \sum_{i}^{N} output_i$$

![over_2]({{ site.url }}/img/dropout.png)


### L2 regularization
가장 일반적으로 사용하는 regularization 기법으로 가중치가 클 수록 큰 패널티를 부과하여 오버피팅을 억제하는 방법입니다. 패널티를 부과하는 방법은 Loss function에 $$ 1/2 \lambda \sum W^2 $$ 을 더해줍니다. 이 값을 미분한 값은 $$ \lambda W$$이고 오차역전파를 통해 계산한 기울기에 $$\lambda W$$를 더하게 되어 가중치 값이 그만큼 보정됩니다. 
참고로 L1 regularization은 loss function에 $$\lambda|W|$$ 을 더해주는 방법입니다. 일반적으로는 L2 regularization을 사용하는 방법이 좋은 결과를 얻을 수 있다고 알려져 있습니다.

