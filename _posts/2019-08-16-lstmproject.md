---
layout: post
title: LSTM을 이용한 데이터 클러스터 분석
excerpt: "LSTM을 이용한 데이터 클러스터 분석"
categories: [project]
comments: true
---
양성자 충돌시 나오는 데이터들은 클러스터를 이루기도 합니다. 원하는 데이터들은 이미 클러스터 안에서 붕괴되었습니다. 하지만 LSTM을 이용하여 클러스터 안에서 붕괴된 입자가 있었는지 분류하고 재구성합니다.

## Introduction

* 유럽입자물리연구소(CERN)에서는 양성자 충돌을 통해 초당 100TB 이상의 데이터를 생성합니다. 
* 가속기에서 생성되는 데이터들은 클러스터를 이루기도 합니다. 
* 클러스터 안에 실제 연구에 사용할 수 있는 데이터가 포함 되어있는지가 중요합니다. 

![project-1]({{ site.url }}/img/project-1.PNG)

### Contents
* `특정입자`에 대한 연구를 위해선 특정 입자가 존재하는 데이터 클러스터를 사용해야 합니다.
* 그러나 대부분의 입자들은 검출기 영역에 도달하기 전에 붕괴합니다.
* 따라서 검출된 클러스터 안의 `구성입자`들의 정보만을 가지고 `붕괴입자`를 찾아 `특정입자`를 재구성 해야합니다.

![project-2]({{ site.url }}/img/project-2.PNG)

### Problem
* 구성입자들 중 붕괴입자가 무엇인지 알 수 없습니다.
* 붕괴입자 하나만으로는 특정입자를 재구성 할 수 없습니다.

## Motivation

* 특정입자를 재구성하는 기존 방법에서는 클러스터 내부의 모든 구성입자에 대해 두개씩 짝지어 모두 재구성 하였습니다.
* 구성입자의 개수 N에 대해 $$ \sum^{N-1}_{i=1} i $$ 번의 계산을 수행하여 많은시간과 불필요한 작업이 수반됩니다.
* 특정입자를 재구성 한 후엔 단순한 cut 조건을 주어 정확도가 떨어집니다.

**$$ \rightarrow $$ Deep Neural Network를 이용하여 계산시간을 효과적으로 단축하고 정확도를 크게 향상 시킬 것으로 기대합니다.**

![project-3]({{ site.url }}/img/project-3.PNG)

## 연구 방법 
### 문제 분류 
##### Classification
* 특정입자가 존재했던 클러스터인지 아닌지 판단하는 `분류문제`

![project-4]({{ site.url }}/img/project-4.PNG)

##### Choice
* 특정입자를 재구성 할 클러스터 내 두개의 구성입자를 찾는 `선택문제`

![project-5]({{ site.url }}/img/project-5.PNG)

### 연구 순서
* **먼저 분류문제를 수행하고 분류문제에서 1을 얻은 클러스터를 대상으로 선택문제를 수행합니다.**

## Data pre-processing
* preprocessing에 관한 모든 코드는 [깃허브](https://github.com/yebiny/RNN_forCluster/tree/master/4-Dataset)에서 참고하실 수 있습니다.

### Number of data 

![project-17]({{ site.url }}/img/project-17.png)

### Input shape

![project-6]({{ site.url }}/img/project-6.PNG)

##### number of particles

![project-16]({{ site.url }}/img/project-16.png)

* Energy ordering을 통해 가장 높은 energy를 가진 구성입자부터 정렬합니다.
* 한 클러스터 안의 구성입자의 개수는 최소 2개부터 최대 50개 이상 까지 존재합니다. 99.9프로 이상의 클러스터는 50개 미만의 구성입자를 가지고 있습니다.
* 인풋 데이터의 사이즈를 고정하기 위해 구성입자 개수가 50 개보다 적으면 0으로 padding 합니다.

##### feature

![project-7]({{ site.url }}/img/project-7.png)

* feature 에 들어가는 구성입자의 정보는 총 여섯가지입니다.
* 그 중 PID는 muon , electron, hadron 총 세가지로 구분됩니다.
* PID는 [one-hot encoding](https://yebiny.github.io/articles/2019-08/onehot)으로 3차원 벡터로 변환합니다.
* 따라서 최종 feature의 size는 8입니다.

### Output shape

* 분류문제와 선택문제의 output shape는 각각의 목적에 맞도록 다르게 주었습니다.

##### 분류문제

![project-7]({{ site.url }}/img/project-8.PNG)

##### 선택문제

![project-7]({{ site.url }}/img/project-9.PNG)

## Model

### 모델 선정
* 한개의 클러스터는 여러개의 구성입자로 이루어져 있고 이는 벡터로 표현할 수 있습니다.
* 구성입자 하나만을 가지고 특정입자를 재구성할 수 없고 구성입자 전체의 정보를 확인해야 합니다.

**$$\rightarrow$$ 이러한 데이터의 특성을 가장 잘 반영할 수 있는 모델로 [LSTM](https://yebiny.github.io/articles/2019-07/lstm)을 사용하기로 하였습니다.**
> [LSTM](https://yebiny.github.io/articles/2019-07/lstm)
LSTM은 연속적인 데이터를 처리하는데 효과적입니다. 이전 데이터를 기억하여 다음 데이터에 반영하기 때문에 클러스터 전체 입자를 고려하여 판단을 내려야 하는 이번 연구목적에 잘 부합하는 모델입니다. 

### 모델 구현
모델 구현 코드는 [깃허브](https://github.com/yebiny/RNN_forCluster/tree/master/5-Model)에 정리해 놓았습니다. 

##### 분류 문제

![project-7]({{ site.url }}/img/project-10.PNG)

##### 선택 문제

![project-7]({{ site.url }}/img/project-11.PNG)

## Results

### Classify

##### Model acc, loss
* 총 epcoch는 20번입니다. 
* 모든 epoch 후 accuracy는 0.839(train),0.824(validation)를 얻었습니다.

![project-7]({{ site.url }}/img/project-14.PNG)

##### Model response
* model response를 시각적으로 표현하엿습니다. 클러스터의 구분이 잘 되고 있음을 볼 수 있습니다.
* Train set은 면적으로, test set은 선으로 표현하였습니다. 둘의 차이가 거의 없어 overfitting이 발생하지 않았다고 볼 수 있습니다.
![project-7]({{ site.url }}/img/project-12.png)

##### ROC Curve
* ROC AUC는 0.907로 분류 성능이 좋은 것을 확인할 수 있습니다.
![project-7]({{ site.url }}/img/project-13.png)


