---
layout: post
title: Over fitting
excerpt: "Over fitting의 개념과 예방"
categories: [deep learning]
comments: true
---

주로 참고한 블로그 글은 다음과 같습니다.
{: .notice}
 
 > [쉽게 읽는 프로그래밍](https://m.blog.naver.com/PostView.nhn?blogId=magnking&logNo=221311273459&proxyReferer=https%3A%2F%2Fwww.google.com%2F)

## Over fitting 이란
한국어로 과적합. 단어에서 알수 있듯이 학습하는 데이터 (training data)에 대해 너무나 잘 학습하는 것을 의미합니다. 그런데 너무나 잘 학습하는게 왜 문제일까요? 

![over_1]({{ site.url }}/img/overfitting.png)

그림에서 처럼 왼쪽은 제대로 학습이 되지 않은 상태, 가운데는 최적의 학습상태, 그리고 오른쪽이 overfitting 상태입니다. 만약 오른쪽 그림처럼 overfitting 인 상태라면 학습 데이터에 대해서만 정확하게 학습되어 실제 데이터에서는 좋은 성능을 보여주지 못하게 됩니다. 따라서, 아래 그림에서 처럼 traing data와 test data에 대해서 error 값이나 accuracy 값이 과하게 차이가 난다면 overfitting이 일어났다고 할 수 있습니다. 
따라서 overfitting이란 학습데이터에 대해 과하게 학습하여 실제 데이터에 대한 오차가 증가하는 현상 이라 할 수 있습니다. 

![over_2]({{ site.url }}/img/overfitting2.png)

## Over fitting 예방책
### validation data
### dropout
드롭아웃은 각 계층마다 일정 비율의 뉴런을 임의로 정해 drop 시켜 나머지 뉴런들만 학습하도록 하는 방법입니다. 
### regularization
