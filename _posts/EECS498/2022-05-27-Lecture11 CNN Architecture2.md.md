---
title : '[EECS498]Lecture11: CNN Architecture'
layout : single
categories: 'EECS498'
tag: [ 'stat', 'CV']
sidebar_main : true
toc : true
date : 2022-04-22
last_modified_at : 2022-05-25
---





# CNN Architecture-2: Recent advanced CNN techniques

------

## 1. Grouped Convolution 



### 1) model

![image-20220525031117399](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525031117399.png)



![image-20220525031216502](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525031216502.png)

Grouped convolution은 Channel을 G개의 그룹으로 나누고, 각각의 그룹에 대해 parallel 하게 conv를 적용해 $C_{out}/G $  의 output채널을 형성한다. 



### 2) why?



![image-20220525031426093](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525031426093.png)



Grouped convolution을 통해 필요한 FLOPs 양을 대폭 줄일 수 있다.



## 2.  ResNeXt



![image-20220525031542109](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525031542109.png)



기존의 Bottleneck Residual Block을 G개의 parallel한 bottleneck block을 사용하는 것으로 변형하였다. 차이점은, C가 아니라 c로 줄인다는 것. 따라서 total FLOPs가 줄어든다. 



![image-20220525031816640](../../../../../../AppData/Roaming/Typora/typora-user-images/image-20220525031816640.png)

이는 앞서 배운 Grouped Convolution을 적용한 것과 비슷하다.



## 3. Squeeze-and-Excitation Networks(SENet)

![image-20220525031953448](../../../../../../AppData/Roaming/Typora/typora-user-images/image-20220525031953448.png)

마지막 ImageNet 대회인 2017년에 우승한 모델이다. 

(2017년이 마지막 imagenet 대회)



## 4. MobileNets

적은 계산으로 어느정도 괜찮은 성능을 보여주는 모델을 만들고자 했다. 



![image-20220525032253495](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525032253495.png)



위와 같이 DepthWise convolution을 적용하여 계산량을 줄였다. 



![image-20220525032436017](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525032436017.png)



ReLU 6, 1x1 conv layer로 시작 등 여러 방법론을 추가한 MobileNetV2도 존재.



## 5. Neural Architecture Search (NAS)

![image-20220525032618576](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525032618576.png)





![image-20220525032711983](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525032711983.png)

모델을 찾아내는 NAS라는 방법론이 개발되었으나, 훈련이 매우 오래걸린다..



## 6. Model Scaling : EfficientNets



![image-20220525032921368](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525032921368.png)

모델의 깊이, 너비, 입력 이미지의 크기를 효율적으로 조정할 수 있는 scaling 방법론이다. 



![image-20220525033009566](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525033009566.png)



성능은 기존의 모델 대비 압도적인 수준임을 확인할 수 있다.  그러나, 여전히 느리다는 한계..



## 7.Beyond NAS

![image-20220525033128024](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525033128024.png)

최근에는 NAS를 넘어서 직접 구조를 찾는 시도가 다시 대세가 되어가고 있다.



## 8. Regnets

![image-20220525033307995](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525033307995.png)



## 9. Summary

![image-20220525033342426](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220525033342426.png)