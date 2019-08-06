---
layout: post
title: Regression
excerpt: "Linear regression"
categories: [machine learning]
comments: true
---

이 포스팅은 `KMOOC 자율주행을위한 머신러닝 강좌`를 청강한 후 정리하였습니다.
{: .notice}

딥러닝을 이해하기 위해서는 Linear Regression과 Logistic Regression을 이해할 필요가 있습니다. 오늘은 우선 선형회귀에 대해서 알아보겠습니다. 선형회귀 개념 자체에 대한 이해도 중요하지만 머신러닝에서 쓰이는 용어인 가설, 손실함수, 경사하강법에 대한 개념까지 함께 이해하도록 하겠습니다.

## Linear regression

시험 공부 시간이 늘어날수록 성적이 잘 나오고 운동시간을 늘릴수록 몸무게는 줄어듭니다. 집의 평수가 클수록 집의 매매 가격은 비싼 경향이 있습니다. 수학적으로 생각해보면 어떤 요인의 수치가 특정 수치에 영향을 주고있습니다. 다른 변수의 값을 변하게 하는 변수를 ``x``, 변수 x에 의해 값이 종속적으로 변하는 변수를 ``y``라고 합니다. 

선형 회귀는 종속 변수 y와 한 개 이상의 독립 변수 x 와의 선형 관계를 모델링하는 분석 기법입니다. 즉 x는 1개일수도, 그 이상일 수도 있습니다. 독립변수 x가 1개일 때 단순 선형 회귀라고 합니다.

### Simple Linear Regression Analysis

집의 크기에 따른 집값을 생각 해 봅시다. 집의 크기를 x, 집값이 y라 하면 수식은 다음과 같이 나타낼 수 있습니다.

* $$y = Wx + b$$ 

위의 수식은 단순 선형 회귀 분석(Simple Linear Regression Analysis)의 수식을 보여줍니다. 여기서 W를 ``가중치(weight)``, b를 ``편향(bias)`` 이라고 합니다.  

### Multiple Linear Regression Analysis

* $$ y = W_1x_1 + W_2x_2 + .... W_nx_n +b $$

잘 생각해보면 집값은 집의 크기 뿐만이 아니라 집의 층수, 집이 지어진 연도, 역과의 거리 등의 요소에 의해 정해 집니다. 이렇게 다수의 요소를 가지고 집의 가격을 예측해 봅시다. y는 여전히 1개이지만 x는 이제 여러개가 되었습니다. 이를 다중 선형 회귀 분석(Multiple Linear Regression Analysis)이라고 합니다.

# Matrix

## Matrix





