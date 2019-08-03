---
layout: post
title: Back Propagation
excerpt: "역전파 이해하기"
categories: [deep learning]
comments: true
---

시작하기 앞서 이번 글은 [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/37406) 의 ``역전파 이해하기`` 글을 공부하고 그 내용을 제 나름대로 풀어 쓴 것임을 밝힙니다.
 
 > [딥 러닝을 이용한 자연어 처리 입문](https://wikidocs.net/37406)
 
 
## Overview
딥러닝에서 역전파를 이해하는 것은 아주 중요합니다. 지난 시간에 순전파를 살펴본 것에 이어 이번에는 역전파를 계산하는 과정을 살펴보도록 하겠습니다. 혹시 순전파에 대한 개념을 참고하시려면 지난글 [딥러닝의 학습과정](https://yebiny.github.io/articles/2019-08/propagation)을 참고하시면 좋습니다. 지난 시간에 이어 저희가 사용하는 네트워크 모델은 다음과 같습니다.

![propa2]({{ site.url }}/img/propa2.PNG) 

순전파가 입력층에서 출력층으로 향한다면 역전파는 반대로 출력층에서 입력층 방향으로 계산하면서 가중치를 업데이트해 갑니다. 먼저 출력층과 은닉층 사이의 가중치 ``U`` 를 업데이트하는 단계를 ``역전파 1단계``, 은닉층과 입력층 사이의 가중치 ``W``를 업데이트 하는 단계를 ``역전파 2단계``라고 합시다. 

## Back Propagation Step 1
우선 역전파 1단계를 진행하겠습니다. 역전파 1단계에서 업데이트해야 할 가중치는 $$U_{11}, U_{21}, U_{12}, U_{22}$$ 총 4개입니다. 네개의 가중치를 업데이트하는 원리는 같기 때문에 먼저 $$U_{11}$$ 에 대해서 계산하겠습니다. $$U_{11}$$를 업데이트 하기 위해 경사하강법을 수행하려면 우리는 $$ \frac{\partial E_{tot}}{\partial U_{11}}$$을 계산해야 합니다. 그런데 이 값을 바로 계산 하기는 힘들기 때문에 ``chain rule``을 사용합니다. Chain rule 식을 쓰기가 헷갈리신다면 아래 그림을 참고하시면 됩니다. 그림에서처럼 $$E_{tot}$$와 $$U_{11}$$ 사이에 존재하는 $$o_1$$과 $$t_1$$ 값을 가지고 chain rule 식을 쓰면 됩니다. 그럼 다음과 같이 풀어 쓸수 있습니다.

* $$ \frac{\partial E_{tot}}{\partial U_{11} } = 
\frac{\partial E_{tot}}{\partial o_1} \times 
\frac{\partial o_1 }{\partial t_1} \times 
\frac{\partial t_1}{\partial U_{11} } $$

이렇게 chain rule로 풀어쓰면 우변의 세 항들을 쉽게 계산할수 있습니다. 세 항들을 차례대로 계산해 봅시다.

#### 첫번째항

$$ \bullet \qquad \frac{\partial E_{tot}}{\partial o_1} $$

$$ E_{tot} = E_1 + E_2 = \frac{1}{2}(y_1-o_1)^2 + \frac{1}{2}(y_2-o_2)^2 $$

$$ \therefore \frac{\partial E_{tot}}{\partial o_1} = -(y_1-o_1) $$

#### 두번째항
sigmoid 함수의 미분값은 $$ f(x) \times (1-f(x)) $$ 입니다. sigmoid 함수의 미분은 기억해 주는것이 좋습니다. 이를 이용하여 두번재 항의 미분을 계산하면 다음과 같습니다.

$$ \bullet \qquad\qquad \frac{\partial o_1}{\partial t_1} $$

$$ o_1 = sigmoid(t_1) $$ 

$$ \therefore \frac{\partial o_1}{\partial t_1} = o_1*(1-o_1) $$

#### 세번째항

$$ \bullet \quad \frac{\partial t_1}{\partial U_{11} } $$

$$ t_1 = U_{11} * h_1 + U_{21} * t_2 $$

$$ \therefore \frac{\partial t_1}{\partial U_{11} } = h_1 $$

#### 최종계산
우변의 모든 항을 계산 하였습니다. 이제 이 값을 모두 곱해주면 됩니다. 최종값의 모든 파라미터들은 상수기 때문에 값만 넣어서 계산하면 됩니다. 계산 결과는 다음과 같습니다.

* $$ \frac{\partial E_{tot}}{\partial U_{11} } = 
\frac{\partial E_{tot}}{\partial o_1} \times 
\frac{\partial o_1 }{\partial t_1} \times 
\frac{\partial t_1}{\partial U_{11} } $$  

$$=-(y_1-o_1) \times o_1(1-o_1) \times h_1 $$

$$= 0.02592286 $$

이제 앞에서 배웠던 경사하강법을 통해 가중치를 업데이트 하면 됩니다. 하이퍼파라미터에 해당하는 learning rate $$\alpha$$는 0.5라고 가정합니다. 그러면 업데이트되는 가중치 $$U_{11}^{+}$$는
* $$U_{11}^{+} = U_{11} - \alpha * \frac{\partial E_{tot}}{\partial U_{11} } $$

$$ = 0.45 - 0.5 * 0.02592286 $$

$$ = 0.43703857 $$

이와 같은 원리로 $$ U_{21}, U_{12}, U_{22} $$ 를 계산할 수 있습니다. 모두 정리하면,
* $$ \frac{\partial E_{tot}}{\partial U_{11} } = 
\frac{\partial E_{tot}}{\partial o_1} \times 
\frac{\partial o_1 }{\partial t_1} \times 
\frac{\partial t_1}{\partial U_{11} } $$

$$\rightarrow U_{11}^{+}=0.43703857$$

* $$ \frac{\partial E_{tot}}{\partial U_{21} } = 
\frac{\partial E_{tot}}{\partial o_1} \times 
\frac{\partial o_1 }{\partial t_1} \times 
\frac{\partial t_1}{\partial U_{21} } $$

$$\rightarrow U_{21}^{+}=0.38685205$$

* $$ \frac{\partial E_{tot}}{\partial U_{12} } = 
\frac{\partial E_{tot}}{\partial o_2} \times 
\frac{\partial o_2 }{\partial t_2} \times 
\frac{\partial t_1}{\partial U_{12} } $$

$$\rightarrow U_{12}^{+}=0.69629578$$

* $$ \frac{\partial E_{tot}}{\partial U_{22} } = 
\frac{\partial E_{tot}}{\partial o_2} \times 
\frac{\partial o_2 }{\partial t_2} \times 
\frac{\partial t_2}{\partial U_{22} } $$

$$\rightarrow U_{22}^{+}=0.59624247$$

## Back Propagation Step 2
1단계를 완료하였다면 이제 입력층 방향으로 이동하며 다시 계산을 이어갑니다. 위의 그림에서 빨간색 화살표는 순전파의 정반대 방향인 역전파의 방향을 보여줍니다. 현재 인공 신경망은 은닉층이 1개밖에 없으므로 이번 단계가 마지막 단계입니다. 하지만 은닉층이 더 많은 경우라면 입력층 방향으로 한 단계씩 계속해서 계산해가야 합니다.

이번 단계에서 계산할 가중치는 $$ W_{11}, W_{21}, W_{12}, W_{22}$$ 입니다. 우선 $$W_{11}$$에 대해서 먼저 업데이트를 진행해보겠습니다. 역전파 1단계에서와 마찬가지로 경사 하강법을 수행하려면 가중치 $$W_{11}$$을 업데이트해야 하고, 업데이트 하기 위해서는 $$\frac{\partial E_{tot}}{\partial W_{11}}$$를 계산해야 합니다. 

위에서와 마찬가지로 $$\frac{\partial E_{tot}}{\partial W_{11}}$$를 계산하기 위해서 chain rule을 사용합니다. 눈치가 빠르신 분들은 역전파 1단계에서의 파라미터들 $$ o_1, t_1, U_{11}, h_1 $$ 가 역전파 2단계에서 각각 $$ h_1, z_1, W_{11}, x_1$$ 로 치환되었다는 것을 알 수 있습니다. 따라서 역전파 1단계에서 chain rule을 그대로 쓰면

* $$ \frac{\partial E_{tot}}{\partial U_{11} } = 
\frac{\partial E_{tot}}{\partial o_1} \times 
\frac{\partial o_1 }{\partial t_1} \times 
\frac{\partial t_1}{\partial U_{11} } $$

여기서  $$ o_1, t_1, U_{11} $$ 을  $$ h_1, z_1, W_{11}$$로 치환합니다. 그러면,

* $$ \frac{\partial E_{tot}}{\partial W_{11} } = 
\frac{\partial E_{tot}}{\partial h_1} \times 
\frac{\partial h_1 }{\partial z_1} \times 
\frac{\partial z_1}{\partial W_{11} } $$

이와 같은 식을 얻을 수 있습니다. 이참에 미분 값도 바로 구해 볼까요? 역전파 1단계에서의 미분계산 결과는 다음과 같았습니다.
* $$ -(y_1-o_1) \times o_1(1-o_1) \times h_1 $$

다만 역전파 1단계와는 달리 우변의 첫번째 항인 $$\frac{\partial E_{tot}}{\partial h_1}$$ 에 대해서 추가적으로 chain rule을 이용하여 계산을 해주어야 합니다. 하지만 두번째 항과 세번째 항의 미분값은 $$o_1$$과 $$h_1$$을 각각 $$h_1$$과 $$x_1$$로 치환하기만 하면 됩니다. 따라서 첫번째 항을 제외한 두번째 세번째 항의 미분 결과를 사용하면 
* $$ \frac{\partial E_{tot}}{\partial h_1} \times h_1(1-h_1) \times x_1 $$

위의 식을 얻을 수 있습니다. 이제 남은건 첫번째 항을 계산하는 일입니다. 차근차근 풀어가 보도록 합시다.

#### 첫번째항

* $$ \frac{\partial E_{tot}}{\partial h_1} $$

이 항은 다음 두 값의 합으로 나타낼 수 있습니다.

$$\frac{\partial E_{tot}}{\partial h_1} = \frac{\partial E_{1}}{\partial h_1} + \frac{\partial E_{2}}{\partial h_1}$$

우변의 두 항에 대해서 각각 chain rule을 사용하여 풀어쓰고 값을 계산해 봅시다.

* $$\frac{\partial E_{1}}{\partial h_1} 
=\frac{\partial E_{1}}{\partial o_1} \times
\frac{\partial o_1}{\partial t_1} \times
\frac{\partial t_1}{\partial h_1}$$

$$E_1 = \frac{1}{2}(y_1-o_1)^2$$

$$o_1 = sigmoid(t_1)$$

$$t_1 = h_1*U_{11} + h_2*U_{21} $$

$$\therefore \frac{\partial E_{1}}{\partial h_1} = −(y_1−o_1) \times o_1(1−o_1) \times U_{11} $$

* $$\frac{\partial E_{2}}{\partial h_1} 
=\frac{\partial E_{2}}{\partial o_2} \times
\frac{\partial o_2}{\partial t_2} \times
\frac{\partial t_2}{\partial h_1} $$

$$E_2 = \frac{1}{2}(y_2-o_2)^2$$

$$o_2 = sigmoid(t_2)$$

$$t_2 = h_1*U_{12} + h_2*U_{22} $$

$$\therefore \frac{\partial E_{2}}{\partial h_1} = −(y_2−o_2) \times o_2(1−o_2) \times U_{12} $$

따라서 첫번째 항을 정리하면 다음과 같이 나타낼 수 있습니다. 
* $$\frac{\partial E_{tot}}{\partial h_1} = \frac{\partial E_{1}}{\partial h_1} + \frac{\partial E_{2}}{\partial h_1}$$

$$ = −(y_1−o_1)*o_1(1−o_1)*U_{11} + −(y_2−o_2)*o_2(1−o_2)*U_{12} $$

$$ = 0.02243370+0.00997311 = 0.03240681$$

첫번째 항을 구했으니 이제 세 항을 모두 곱하기만 하면 됩니다. 두번째항과 세번째항은 위에서 구했던 미분 결과값을 쓰겠습니다.

* $$ \frac{\partial E_{tot}}{\partial W_{11} } = 
\frac{\partial E_{tot}}{\partial h_1} \times 
\frac{\partial h_1 }{\partial z_1} \times 
\frac{\partial z_1}{\partial W_{11} } $$

$$ = 0.03240681 \times h_1(1-h_1) \times x_1  $$

$$ = 0.03240681×0.24960043×0.1=0.00080888$$

경사하강법을 통해 가중치를 업데이트 합니다.
* $$W_{11}^{+} = W_{11} - \alpha * \frac{\partial E_{tot}}{\partial W_{11} } $$

$$ = 0.2 - 0.5 * 0.00080888 $$

$$ = 0.29959556 $$

이와 같은 원리로 $$ W_{21}, W_{12}, W_{22} $$ 를 계산할 수 있습니다. 모두 정리하면,

* $$ \frac{\partial E_{tot}}{\partial W_{11} } = 
\frac{\partial E_{tot}}{\partial h_1} \times 
\frac{\partial h_1 }{\partial z_1} \times 
\frac{\partial z_1}{\partial W_{11} } $$

$$ \rightarrow W_{11}^{+}=0.29959556 $$

* $$ \frac{\partial E_{tot}}{\partial W_{21} } = 
\frac{\partial E_{tot}}{\partial h_1} \times 
\frac{\partial h_1 }{\partial z_1} \times 
\frac{\partial z_1}{\partial W_{21} } $$

$$ \rightarrow W_{21}^{+}=0.24919112 $$

* $$ \frac{\partial E_{tot}}{\partial W_{12} } = 
\frac{\partial E_{tot}}{\partial h_2} \times 
\frac{\partial h_2 }{\partial z_2} \times 
\frac{\partial z_1}{\partial W_{12} } $$

$$\rightarrow W_{12}^{+}=0.39964496 $$

* $$ \frac{\partial E_{tot}}{\partial W_{22} } = 
\frac{\partial E_{tot}}{\partial h_2} \times 
\frac{\partial h_2 }{\partial z_2} \times 
\frac{\partial z_2}{\partial W_{22} } $$

$$\rightarrow W_{22}^{+}=0.34928991 $$


## 결과 확인

역전파를 통해서 우리는 모든 가중치 ``U``와 ``W``에 대해서 업데이트를 진행하였습니다. 이제 이 업데이트된 가중치를 이용하여 순전파를 진행하고, 오차가 감소하였는지 살펴보겠습니다.

#### 은닉층
* $$ z_1 = W_{11}^{+}*x_1 + W_{21}^{+}*x_2 = 0.07979778$$
* $$ z_2 = W_{12}^{+}*x_1 + W_{22}^{+}*x_2 = 0.10982248 $$
* $$ h_1 = sigmoid(z_1) = 0.51993887 $$
* $$ h_2 = sigmoid(z_2) = 0.52742806 $$

#### 출력층
* $$ t_1 = U_{11}^{+}*h_1 + U_{21}^{+}*t_2 = 0.43126996 $$
* $$ t_2 = U_{12}^{+}*h_1 + U_{22}^{+}*t_2 = 0.67650625 $$
* $$ o_1 = sigmoid(t_1) = 0.60617688 $$
* $$ o_2 = sigmoid(t_2) = 0.66295848 $$

#### 오차
지난번과 마찬가지로 ``MSE``를 사용합니다.
* $$ E_1 = \frac{1}{2}(y_1-o_1)^2 = 0.02125445 $$
* $$ E_2 = \frac{1}{2}(y_2-o_2)^2 = 0.00198189 $$
* $$ E_{tot} = E_1 + E_2 = 0.02323634 $$

지난 시간에서 구했던 전체 오차 $$E_{tot}$$가 0.02397190였으므로 한번의 역전파를 통해 오차가 감소한 것을 확인할 수 있습니다. 이렇듯 인공 신경망의 학습은 오차를 최소화하는 가중치를 찾는 목적으로 순전파와 역전파를 반복하는 것을 말합니다.
