---
layout: post
title: LSTM을 이용한 데이터 클러스터 분석
excerpt: "LSTM을 이용한 데이터 클러스터 분석"
categories: [project]
comments: true
---

## 데이터 소개

## Introduction

### Point
* `특정입자`에 대한 연구를 위해선 특정 입자가 존재하는 데이터 클러스터를 사용해야 합니다.
* 그러나 대부분의 입자들은 검출기 영역에 도달하기 전에 붕괴합니다.
* 따라서 검출된 클러스터 안의 `구성입자`들의 정보만을 가지고 `붕괴입자`를 찾아 `특정입자`를 재구성 해야합니다.

![project-2]({{ site.url }}/img/project-2.PNG)

### Problem
* 구성입자들 중 붕괴입자가 무엇인지 알 수 없습니다.
* 붕괴입자 하나만으로는 특정입자를 재구성 할 수 없습니다.

##  Motivation
* 특정입자를 재구성하는 기존 방법에서는 클러스터 내부의 모든 구성입자에 대해 두개씩 짝지어 모두 재구성 하였습니다.
* 구성입자의 개수 N에 대해 $$ \sum^{N-1}_{i=1} i $$ 번의 계산을 수행하여 많은시간과 불필요한 작업이 수반됩니다.
* 특정입자를 재구성 한 후엔 단순한 cut 조건을 주어 정확도가 떨어집니다.

**$$ \rightarrow $$ Deep Neural Network를 이용하여 계산시간을 효과적으로 단축하고 정확도를 크게 향상 시킬 것으로 기대합니다.**

![project-3]({{ site.url }}/img/project-3.PNG)


## 연구 방법



#### 문제 분류 
* 특정입자가 존재했던 클러스터인지 아닌지 판단하는 분류문제 -> Classify

![project-4]({{ site.url }}/img/project-4.PNG)

* 특정입자를 재구성 할 클러스터 내 두개의 구성입자를 찾는 선택문제 -> Choice

![project-5]({{ site.url }}/img/project-5.PNG)

## Data pre processing

### Data format 
### Unbalnced data

### Input shape

![project-6]({{ site.url }}/img/project-6.PNG)

#### 구성입자 개수

* 구성입자의 개수는 2개부터 최대 50개 까지 
* 50 개 이하이면 0으로 패딩
* energy ordering

#### feature

![project-7]({{ site.url }}/img/project-7.png)

#### one-hot encoding

PID는 one-hot-encoding을 사용하였습니다.

따라서 총 feature의 개수는 8개 

### Output shape

* 분류문제

![project-7]({{ site.url }}/img/project-8.PNG)

* 선택문제

![project-7]({{ site.url }}/img/project-9.PNG)


## Model

### Data feature
* 한개의 클러스터는 여러개의 구성입자로 이루어져 있고 이는 벡터로 표현할 수 있습니다.
* 구성입자 하나만을 가지고 특정입자를 재구성할 수 없고 구성입자 전체의 정보를 확인해야 합니다.

**$$\rightarrow$$ 이러한 데이터의 특성을 가장 잘 반영할 수 있는 모델로 `LSTM`을 사용하기로 하였습니다**
> [LSTM](https://wikidocs.net/37406)
* LSTM은 연속적인 데이터를 처리하는데 효과적입니다. 이전 데이터를 기억하여 다음 데이터에 반영하기 때문에 클러스터 전체 입자를 고려하여 판단을 내려야 하는 이번 연구목적에 잘 부합하는 모델입니다. 


### Model development
모델 구현 코드는 [깃허브]에 정리해 놓았습니다. [깃허브](https://github.com/yebiny/RNN_forCluster/tree/master/5-Model
)
#### Classify

![project-7]({{ site.url }}/img/project-10.PNG)

#### Choice

![project-7]({{ site.url }}/img/project-11.PNG)

## Results
![project-7]({{ site.url }}/img/project-12.png)

![project-7]({{ site.url }}/img/project-13.png)

![project-7]({{ site.url }}/img/project-14.PNG)

