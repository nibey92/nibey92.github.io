---
layout: post
title: Loss function and optimizer
excerpt: "손실함수와 옵티마이저의 개념과 종류를 알아보자"
categories: [deep learning]
comments: true
---

주로 참고한 블로그 글은 다음과 같습니다.
{: .notice}
 
 > [UMBUM](https://umbum.tistory.com/222)

## Loss function(손실 함수)
손실함수는 실제값과 예측값의 차이를 수치화해주는 함수입니다. 이 두값의 차이, 즉 오차가 클 수록 함수의 값은 크고 오차가 작을 수록 손실함수의 값은 작아집니다. 회귀에서는 `mean squared error`, 분류 문제에서는 `cross entropy`를 주로 손실함수로 사용합니다. 손실함수의 값을 최소화하는 두 개의 매개변수인 가중치 W와 편향 b를 찾아가는 것이 딥러닝의 학습과정이므로 손실함수를 적절히 선정하는 것은 매우 중요한 일입니다.

### Mean Squared Error(MSE)
오차 제곱 평균을 의미합니다. 연속형 변수를 예측할 때 사용됩니다.
* $$-1/N \sum (y-\hat{y})^2 $$

### Cross Entropy Error 

* $$-\sum ylog\hat{y} $$
$$y$$: 실제값 (0 or 1) 
$$\hat{y}$$ : 예측값

낮은 확률로 예측해서 맞추거나 높은 확률로 예측해서 틀리는 경우 loss가 더 큽니다. 이진 분류(Binary classification)의 경우 binary_crossentropy를 사용하며 다중 클래스 분류(Multi-class classification)의 경우  categorical_crossentropy를 사용합니다. 

