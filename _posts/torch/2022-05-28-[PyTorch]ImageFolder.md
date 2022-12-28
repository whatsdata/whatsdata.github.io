---
title: '[PyTorch] ImageFolder'
categories : 'torch'
tag: [ 'stat', 'pytorch']
sidebar_main : true
toc : true
date: 2021-10-03
---

ImageFolder는  PyTorch에서 제공하는 라이브러리로, 로컬에 저장된 데이터셋을 학습에 사용할 때 사용된다.

이는 다음과 같은 계층적인 폴더 구조를 가지고 있는 데이터셋을 불러올 때 사용할 수 있다. 다시 말해 다음과 같이 각 이미지들이 자신의 레이블(Label) 이름으로 된 폴더 안에 들어가 있는 구조라면, ImageFolder 라이브러리를 이용하여 이를 바로 불러와 객체로 만들면 된다.

 

```
dataset/
	0/
		0.jpg
		1.jpg
        	...
	1/
		0.jpg
		1.jpg
		...
	...
	9/
		0.jpg
		1.jpg
		...
```

 

예시로 DCGAN Tutorial에 사용되는 코드를 보자면,



```
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
```

위와 같다.



특이사항이라면, 데이터를 불러올 때 transform의 적용이 가능하다. 

size가 크다면 resize를 하고,

중간만 따로 파내고 싶다면 CenterCrop

... 이와 같다.



동 튜토리얼에서 추가적인 내용을 보자면, 



```
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
```



DataLoader 함수를 통해 데이터를 iterable하게 불러오는 함수를 설정하고, 몇가지 그래프를 그려보면 

![image-20220528032230539](https://raw.githubusercontent.com/whatsdata/whatsdata.github.io/master/img/2022-05/image-20220528032230539.png)



이와 같이 Local의 데이터를 불러와서 그림으로 표현할 수 있다!