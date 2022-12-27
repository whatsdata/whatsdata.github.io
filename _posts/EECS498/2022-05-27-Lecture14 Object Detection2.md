---
title : '[EECS498]Lecture14: Object detection2'
layout : single
toc : true
categories: 'EECS498'
tag: [ 'CV']
date : 2022-05-25
last_modified_at : 2022-05-27
---





## 3. 

![image-20220527084847980](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527084847980.png){: .align-center}

앞서 배운 R-CNN은 느리다는 문제를 가진다. 2000개 정도의 proposal을 뽑아냈다면, <span style="color:blue">이들에 대해 2000번의 forward pass를 해야한다!</span>sapn> 모든 경우의 수를 고려하는 것보다는 빠르지만, 절대 이 방법이 빠르다고 할 수는 없다. 



이에 Fast R-CNN 이라는 시간단축 방법을 배운다.



### 3.1. 설명

![image-20220527085401056](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527085401056.png){: .align-center}

- 기존의 R-CNN은 region proposals 각각을 CNN에 넣어서 시간이 오래걸렸다. 

  ![image-20220527085512608](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527085512608.png){: .align-center}

- 이와는 달리, fast R-CNN은 일단 이미지 전체에 대해 CNN을 적용해 Feature에 대한 정보를 얻는다.

  이때 fc layer는 없이 넣어서 feature map을 얻는다.

  <img src="https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527085632024.png" alt="image-20220527085632024" style="zoom:90%;" />{: .align-center}

  

- 위에서 얻은 feature map에서 region proposal들을 crop&resize 하여 cnn을 한다. 그리고 이를 통해 score를 얻는다.

- 이 방식으로 하면, backbone network의 계산을 통해 반복적인 계산을 많이 줄여서 효율적으로 된다. 



### 3.2. How to Crops? : Rol Pool

![image-20220527093631705](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527093631705.png){: .align-center}

- Receptive fields 개념으로 input image에 해당하는 image features를 찾을 수 있다.

  그러나 이 경우, feature grid와 정확하게 일치하지 않을 수 있다. 따라서 이를 조절하는데, 

  ![image-20220527093826956](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527093826956.png){: .align-center}

  이를 snap이라 칭한다.

![image-20220527093912433](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527093912433.png){:.align-center}

그리고 해당 이미지를 대강의 2x2 사이즈로 나눈 후, maxpool을 적용한다 

이를 통해 동일한 사이즈의 region을 뽑아낼 수 있다. (resize)



### 3.3. How to Crops?: Rol Align

Rol pool의 경우 snapping 과정에서 약간의 misalignment가 존재한다. 

이를 해결하기 위한 Rol Align 방식이 있으나, 생략. 



### 3.4.  speed

![image-20220527094309458](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527094309458.png)



fast R-CNN은 R-CC보다 훨씬 빠르지만, Region proposal에서 시간이 여전히 걸린다.

(R-CNN은 상대적으로 그 전 단계인 feature 계산 부분에서 시간이 오래걸려서 티가 안나지만..)

따라서 이를 줄이기 위한 또 다른 방식으로 Learnabel Region Proposal을 소개한다.



### 3.5. Fast***er*** R-CNN: Learnable Region Proposals

![image-20220527094622535](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220527094622535.png)



여기서는 fast R-CNN에다가 `Region Proposal Network` 라는 레이어를 하나 더 추가한다. 

이는 region Proposals의 prediction을 도와주는 역할을 하는데, 

backbone network를 통과시켜 나온 feature map을 `RPN`에 통과시켜 region proposals를 predict하고, 뽑은 region proposals를 가지고 Fast R-CNN에서 했던 것과 모두 같은 방식으로 나머지 과정을 처리한다.

Fast R-CNN과의 차이는 `RPN`밖에 없다.

