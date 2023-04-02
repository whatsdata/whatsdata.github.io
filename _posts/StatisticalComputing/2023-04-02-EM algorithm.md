---
layout : single
title : '[Statistical Computing] EM algorithm'
categories : 'StatisticalComputing'
sidebar_main : true
tag : ['stat', 'EM algorithm']
toc : true
date: 2023-04-02
last_modified_at : 2023-04-02
---

><b>References </b>
>
>- Yonsei univ. STA6171 : Statistical Computing



- EM algorithm은 본래 결측치 처리 문제(Missing data problem)를 처리하기 위해 개발되었지만, 결측치 처리가 아닌 일반적인 문제에도 일반화될 수 있다. 
- 결측치가 없는 경우에도, 결측치가 있는 것처럼 행동하여 EM algorithm을 적용할 수 있다.




### EM algorithm

- Parameter를 알고 있다면, 결측치 데이터를 채우는 것은 쉽다.

- 모든 데이터에 대한 정보가 있다면, Parameter를 추정하는 것은 쉽다. 



- 따라서, E step과 M step을 번갈아가면서 문제를 푼다.
  - E step  :  the expectation step (Data imputation)
  - M step : the maximization step (Parmeter estimation)






- Steps
	1. Start with an initial guess for parameters , $\theta^{(0)}$.
	2. Iterate between,
		- E-step : Compute $$ Q(\theta \vert \theta^{(t)}) = E \lbrack l(\theta \vert y_{obs} , y_{mis} ) \vert y_{obs} , \theta^{(t)} \rbrack  $$
		- M-step : Find $\theta^{(t+1)}$ such that $$ \theta^{(t+1)} = argmax_{\theta} \,\, Q(\theta \vert \theta^{(t)})$$






- 위의 steps를 따라가면 EM algorithm을 통해 Log-likelihood function을 극대화하는 $\theta$에 근접할수 있다는 것은 다음의 Ascent property를 통해 알 수 있다.






- <font color="#ff0000">Ascent property</font> : For a sequence $\lbrace \theta^{(0)}, \theta^{(1)},\cdots \rbrace$ , we have 


$$
l(\theta^{(t+1)} \vert y_{obs}) \geq l(\theta^{(t)} \vert y_{obs})
$$


- 즉, Observed data의 Log-likelihood를 maximize하는 추정치를 얻을 수 있다.



**Proof)**

- Complete data likelihood는 다음과 같이 분해된다.

  $$p(y_{com} \vert \theta) = p(y_{obs}, y_{mis} \vert \theta) = p(y_{obs})$$



- 여기에 Log를 씌우면 다음과 같다. 

  $$ l(\theta \vert y_{obs}) = l( \theta \vert y_{com}) - log \big( p(y_{mis} )\vert y_{obs}, \theta )  \big) $$