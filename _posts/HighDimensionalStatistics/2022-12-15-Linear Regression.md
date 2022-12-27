---
layout : single
title : '[High Dimensional Statistics] Linear Regression'
categories : 'HighDimensionalStatistics'

tag : ['Linear Regression']
toc : true
date: 2022-11-26
last_modified_at : 2022-12-15

---

><b>References </b>
>
>- Yonsei univ. STA9104 : Statistical theory for high dimensional and big data 
>- High-Dimensional Statistics : A Non-Asymptotic Viewpoint, 2017, Martin J. Wainwright



Goal : Develop linear regression problem in high-dimensional setting. Linear regression has long history of success, but we need to develop set up for high-dimensional setting, $e.g. ~d \geq n $.

<Br>

## 1. Problem set up

><b> notation </b>
>
>- $\theta^\star \in \mathbb{R}^d$  be an unknown regression vector.
>- $Y =( Y_1 , \cdots , Y_n)^T \in \mathbb{R}^n$ and fixed design matrix $X \in \mathbb{R}^{n \times d}$
>- $\epsilon = (\epsilon_1 , \cdots , \epsilon_n )^T \in \mathbb{R}^n$ and each $\epsilon_i$s are independent centered <b> sub-Gaussian</b> random variables with variance proxy $\sigma^2$



With the notations, we define linear regression problem like


$$
Y = X \theta^\star + \epsilon
$$


Given above setting, the aim of Linear regressions are like : 



<b> 1.1. Prediction</b>

Let $\tilde{Y} = X\theta^\star + \tilde{\epsilon}$ to be an independent draw, which is identically distributed as $Y$. Then, estimator for performance is :


$$
\frac{1}{n} \mathbb{E} [ \vert \vert \tilde{Y} - X \hat{\theta}\vert \vert_2^2] =\frac{1}{n} \mathbb{E} [ \vert \vert \tilde{\epsilon} + X( \theta^\star - \hat{\theta})\vert \vert_2^2] = \underset{unavoidable ~~error}{\frac{1}{n} \mathbb{E}[\vert \vert \tilde{\epsilon}\vert \vert_2^2]} + \underset{mean ~~squared ~~error}{\frac{1}{n} \mathbb{E} [\vert \vert X( \theta^\star - \hat{\theta})\vert \vert_2^2]}
$$


 <B>1.2. Parameter estimation</B>

Sometimes, only on parameter estimation : 


$$
\mathbb{E}[ \vert  \vert \theta^\star - \hat{\theta}\vert \vert_2 ^2]
$$
<br>

## 2. Least squares estimator in high dimensions

- well-known OLS estimator is such that :

$$
\hat{\theta}_{LS} = (X'X)^{-1}X'Y = arg \underset{\theta}{min} \vert \vert Y-X \theta \vert \vert_2^2
$$

- However, in High-dimensional setting with $d \geq n$, $(X'X)$ is not invertible. Instead, one can get normal equation, 

$$
X'X \theta = X'Y
$$

​	, from which one can get


$$
\hat{\theta}_{LS} = (X'X)^{-}X'Y
$$

---

<b> Definition of Generalized Inverse </b>

For the linear transformation $A : R^d \rightarrow R^n$, A generalized inverse of $A$ is the linear transformation $A^-$ such that  


$$
AA^- y = y ~~for ~~all ~~y \in C(A)
$$


 Apply above definition to $\beta = X^{-} Y $ and $X \beta = Y$, 


$$
X \beta = XX^- Y = Y ~~for ~~all~~Y \in C(X)
$$


Therefore, $\beta = (X'X)^- X'Y$ is the solution to the normal equation, although it does not need to be unique.

---



- For more information about OLS with generalized inverse, one can check 'estimation' post in linear model category of this blog.



### 2.1. Mean square error of the least square estimator



<b> Theorem (Least squares estimators) </b> Assume that the linear model $Y= X \theta^\star +\epsilon$ satisfies the above notations. Then, next works:


$$
\frac{1}{n} \mathbb{E} [\vert \vert X( \theta^\star - \hat{\theta})\vert \vert_2^2] \leq \sigma^2 \frac{r}{n}
$$


where $r$ is the rank of $X'X$. Moreover, for any $\delta >0$, with probability at least $1-\delta$, it holds


$$
\frac{1}{n} \mathbb{E} [\vert \vert X( \theta^\star - \hat{\theta})\vert \vert_2^2] \leq \sigma^2 \frac{r+ log(1/\delta)}{n}
$$


Which states that if $\frac{r}{n} \rightarrow 0$, mean squared error goes to zero



<b> proof)</b> proof consists of two steps, step 1) *Basic inequality*, step 2) *sup-out technique*



*Step 1) Basic inequality*



By the definition of $\hat{\theta}$, 


$$
\vert \vert Y- X \hat{\theta} \vert \vert _2 ^2 \leq \vert \vert Y- X \theta^\star \vert \vert _2 ^2 = \vert  \vert \epsilon \vert \vert_2^2
$$


Moreover,


$$
\vert \vert Y-X \hat{\theta}_{LS} \vert \vert _2^2 = \vert \vert \epsilon +X(\theta^\star -  \hat{\theta}_{LS}) \vert \vert _2^2 = \vert \vert \epsilon \vert \vert_2^2 + 2\epsilon ^T X(\theta^* - \hat{\theta}_{LS}) + \vert \vert X(\theta^\star -\hat{\theta_{LS}})\vert \vert_2 ^2
$$


Combining two inequalities yields the basic inequality:


$$
\vert \vert X( \theta^\star - \hat{\theta}_{LS} )\vert\vert_2^2 \leq 2\epsilon ^T X( \hat{\theta}_{LS}- \theta^* ) = 2 \vert \vert X(\hat{\theta}_{LS} - \theta ^\star)\vert\vert_2 \times \frac{\epsilon ^T X( \hat{\theta}_{LS}- \theta^* )}{\vert \vert X(\hat{\theta}_{LS} - \theta ^\star)\vert\vert_2}
$$


Because  $\epsilon \,\, \& \,\, \hat{\theta}_{LS} $  are dependent , right part of last term is hard to control. Therefore, Sup-out technique is applied .

<Br>

*Step 2) Sup - out Technique*



Let $\Psi = (\psi_1 , \cdots , \psi_r) \in \mathbb{R}^{n \times r}$ is an orthonormal basis for $col(X)$, which is,


$$
X( \hat{\theta} - \theta^{\star}) = \Psi \nu ~~ for ~~ \nu \in \mathbb{R}^r
$$


It yield : 


$$
\begin{align}

\frac{\epsilon ^T X( \hat{\theta}_{LS}- \theta^* )}{\vert \vert X(\hat{\theta}_{LS} - \theta ^\star)\vert\vert_2} &= \frac{\epsilon^T \Psi \nu}{\vert \vert \Psi \nu \vert \vert _2} \\
&=\frac{\tilde{\epsilon}^T \nu}{\vert \vert \nu \vert \vert_2} \quad \leftarrow \quad \vert \vert \Psi \nu \vert \vert _2 = \sqrt{\nu ^T \Psi^T \Psi \nu} = \vert \vert \nu \vert \vert_2\, , \, \epsilon^T \Psi \equiv \tilde{\epsilon}^T \\
&\leq \underset{u\in \mathbb{R}^r : \vert\vert u \vert \vert_2 =1}{sup} \tilde{\epsilon}^T u

\end{align}
$$


Therefore, 
$$
\vert \vert X(\hat{\theta}_{LS} - \theta ^\star)\vert\vert_2^2 \leq 4 \underset{u\in \mathbb{R}^r : \vert\vert u \vert \vert_2 =1}{sup} (\tilde{\epsilon}^T u)^2
$$
<BR>

Now, Check the claim : $\tilde{\epsilon} $ ~ $sub- Gaussian (\sigma^2)$ to bound above equation. 



using the fact that $E[\tilde{\epsilon} ]=0 $.


$$
\forall u \in S^{r-1} , \quad E[e^{s \tilde{\epsilon}^Tu}] = E[e^{s \epsilon^T \Psi u}] \leq e^{\frac{s^2 \sigma^2}{2}}
$$


It satisfies the definition of sub-gaussian ($E[e^{\lambda (X-E[X]) }]\leq e^{\frac{\sigma^2 \lambda^2}{2}}  $ ), meaning that $\tilde{\epsilon}$ is sub-gaussian



Then, By cauchy - schwarz inequality($ \vert \langle u,v \rangle \vert^2  \leq \langle u,u \rangle \cdot \langle v,v\rangle $),


$$
\mathbb{E} \Bigg[ \underset{u\in \mathbb{R}^r : \vert\vert u \vert \vert_2 =1}{sup} (\tilde{\epsilon}^T u)^2 \Bigg] \leq \sum_{i=1}^r \mathbb{E}[\tilde{\epsilon}_i^2] \leq r \sigma^2
$$


Therefore, 


$$
\frac{1}{n} \mathbb{E} [\vert \vert X( \theta^\star - \hat{\theta})\vert \vert_2^2] \leq \mathbb{E} \bigg[ \frac{4}{n}  \underset{u\in \mathbb{R}^r : \vert\vert u \vert \vert_2 =1}{sup}  (\tilde{\epsilon}'u)^2 \bigg] \leq \frac{4r}{n} \sigma^2
$$


Then, by the fact that $\underset{\theta \in \mathbb{E}}{sup} \,\, \theta^T X \leq \,\, 2 \underset{z \in N_{1/2}}{max} z^TX$ from Metric Entropy, 


$$
\begin{align}

\mathbb{P} (\vert \vert X( \theta^\star -\hat{\theta}_{LS})  \vert \vert _2^2 \geq t) & \leq \mathbb{P} \Big( \underset{u \in N_{1/2}}{max} (\tilde{\epsilon}^T u)^2 \geq \frac{t}{16} \Big)  \\

& \leq \sum _{u \in N_{1/2} }{\mathbb{P} (\tilde{\epsilon}^T u \geq \frac{t}{16})} \\
& \leq \vert N_{1/2} \vert e^{ - \frac{t}{32 \sigma^2}} ~~\longleftarrow ~chernoff ~~ bound \\
& \leq 5^r e^{ - \frac{t}{32 \sigma^2}} ~~\,~\quad~\longleftarrow ~property~~of~~packing



\end{align}
$$


Now, consider $t$ such that 
$$
5^r e^{- \frac{t}{32 \sigma^2}} \leq \delta \longleftrightarrow t \geq 32 \sigma^2 \{ r log5 + log(1/ \delta) \}
$$
we can  conclude that 
$$
\frac{1}{n} \mathbb{E} [\vert \vert X( \theta^\star - \hat{\theta})\vert \vert_2^2] \lesssim \sigma^2 \frac{r+ log(1/\delta)}{n}
$$
with at least probability $\delta$.

<br>

## 3. Sparse linear regression

