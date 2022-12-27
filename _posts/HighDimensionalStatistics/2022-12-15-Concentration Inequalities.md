---
layout : single
title : '[High Dimensional Statistics] Concentration Inequalities'
categories : 'HighDimensionalStatistics'

tag : ['Inequalities']
toc : true
date: 2022-09-15
last_modified_at : 2022-10-25
---

><b>References </b>
>
>- Yonsei univ. STA9104 : Statistical theory for high dimensional and big data 
>- High-Dimensional Statistics : A Non-Asymptotic Viewpoint, 2017, Martin J. Wainwright



## 0. Motivation



## 1. From Markov to Chernoff



### 1.1. Markov's Inequality



### 1.2. Chebyshev's Inequality



### 1.3. Polynomial Markov



### 1.4. Chernoff bound



## 2. Sub-Gaussian random variables

### 2.1. Definition



### 2.2. Properties



### 2.3. Equivalent Definitions



### 2.4. Sub-Gaussian random vectors



### 2.5. Hoeffding's inequality



### 2.6. Maximal Inequality



## 3. Sub-Exponential random variables

### 3.1. Definition



### 3.2. examples



### 3.3. Bernstein's Condition



### 3.4. The Johnson-Lindenstrauss Lemma





## 4. Bounded difference conditions



​	One fact about concentration is that if we change the value of one random variable, the function does not change dramatically.

​	Formally, we have independent random variables $X_1, \cdots,X_n,$ where each $X_i \in \mathbb{R}$. We have a function $f : \mathbb{R}^n \rightarrow \mathbb{R}$, that satisfied the property that : 
$$
\vert f(x_1, \cdots, x_{k-1}, x_k, x_{k+1}, \cdots , x_n) - f(x_1, \cdots, x_{k-1}, x_k', x_{k+1}, \cdots , x_n) \vert \leq L_k
$$


for every $x,x' \in \mathbb{R}^n$. This is known as the <i> Bounded difference condition </i>

<Br>

### 4.1. McDiarmid's inequality

<b> Theorem (McDiarmid's inequality)</b> <i> If the random variables $X_1, \cdots , X_n$ are independent and a function f : $\mathbb{R}^n \rightarrow \mathbb{R}$ satisfies the bounded difference condition above, then for all </i>$t \geq 0 $ , 


$$
\mathbb{P} (\vert f(X_1, \cdots ,X_n ) - \mathbb{E} [f(X_1, \cdots , X_n) ]\vert \geq t)  \leq 2 exp \bigg( - \frac{2t^2} {\sum_{k=1}^n L_k^2}\bigg)
$$


<b> proof) </b> 

<B>Claim</b> : Let $V_i = E[f \vert X_1 , \cdots, X_i ] - E[ f \vert X_1, \cdots, X_{i-1}]$ , then  $E[ e^{\lambda V_i} \vert X_1, \cdots ,X_{i-1}]$ = $e^{\frac{\lambda^2 L_i^2 }{8}}$ 



Let's notate,



$ A_i = \underset{x}{sup} \{ E[f \vert X_1, \cdots ,X_{i-1}, x] - E[f\vert X_1, \cdots ,X_{i-1}]  \}$

 

$B_i = \underset{x}{inf} \{ E[f\vert X_1, \cdots ,X_{i-1}, x] - E[f\vert X_1, \cdots ,X_{i-1}]  \} $





Since $B_i \leq V_i \leq A_i$, by Hoeffding's lemma and the fact that $A_i - B_i \leq L_i$,

---



1. <b>Hoeffding's Lemma</b> : If $a \geq X_i \geq b$, $\mathbb{E} [e^{\lambda (X_i -\mu) }]\leq e^{\frac{\lambda^2 (b-a)^2}{8}} \quad for\quad \lambda \in \mathbb{R^+}$	



2. $A_i - B_i \leq L_i$


$$
\begin{align} 

pf )  \quad A_i - B_i  &= \underset{x}{sup }\{ E[f| X_1, \cdots ,X_{i-1}, x] - \underset{x}{inf} \{ E[f| X_1, \cdots ,X_{i-1},x]  \}\\
& \leq \underset{x,y}{sup} \vert E[f| X_1, \cdots ,X_{i-1}, x] -  E[f| X_1, \cdots ,X_{i-1},y] \vert \\
& = \underset{x,y}{sup} \vert E_{X_{i+1}, \cdots, X_n}[f| X_1, \cdots ,X_{i-1}, x, X_{i+1}, \cdots ,X_n] \\
&\quad\quad\quad\quad\quad\quad \, \,\,- \,\,[f| X_1, \cdots ,X_{i-1},y,X_{i+1}, \cdots ,X_n] \vert \\
& \leq \underset{x,y}{sup} \vert E_{X_{i+1}, \cdots, X_n} \vert [f| X_1, \cdots ,X_{i-1}, x, X_{i+1}, \cdots ,X_n] \\
&\quad\quad\quad\quad\quad\quad \, \,\,- \,\,[f| X_1, \cdots ,X_{i-1},y,X_{i+1}, \cdots ,X_n] \vert \vert \\
& = \underset{x,y}{sup}  E_{X_{i+1}, \cdots, X_n} \vert [f| X_1, \cdots ,X_{i-1}, x, X_{i+1}, \cdots ,X_n] \\
&\quad\quad\quad\quad\quad\quad \, \,\,- \,\,[f| X_1, \cdots ,X_{i-1},y,X_{i+1}, \cdots ,X_n] \vert  \\
&\leq L_i

\end{align}
$$

---



Therfore,  $E[ e^{\lambda V_i} \vert X_1, \cdots ,X_{i-1}]$ = $e^{\frac{\lambda^2 L_i^2 }{8}}$

<br>

Then, 


$$
\begin{align}

\mathbb{P} (\vert f(X_1, \cdots ,X_n ) - \mathbb{E} [f(X_1, \cdots , X_n) ]\vert &= \mathbb{P} \bigg(  \sum_{i=1}^n V_i \geq t \bigg)\\
&= \mathbb{P} \bigg( e^{\lambda   \sum_{i=1}^n V_i }\geq e^{\lambda t} \bigg)\\
& \leq e^{-\lambda t}\mathbb{E} \big[ e^{\lambda \sum_{i=1}^n V_i }\big]\quad \longleftarrow \quad Markov \,\, Inequality \\
& = e^{-\lambda t} \mathbb{E} \big[e^{\lambda \sum_{i=1}^{n-1} V_i} \mathbb{E}\big[ e^{\lambda V_n} \vert X_1, \cdots, X_{n-1}  \big]\big] \quad \\
&\leq \cdots \\
&\leq e^{-\lambda t} e^{\lambda^2 \sum_{i=1}^n L_i^2 /8} \\& = exp \bigg( - \frac{2t^2} {\sum_{k=1}^n L_k^2}\bigg) \longleftarrow \lambda = \frac{4t}{\sum_{i=1}^n L_i^2}

\end{align}
$$


Applying same procedure to the other direction, one can get  $\mathbb{P} (\vert f(X_1, \cdots ,X_n ) - \mathbb{E} [f(X_1, \cdots , X_n) ]\vert \geq t)  \leq 2 exp \bigg( - \frac{2t^2} {\sum_{k=1}^n L_k^2}\bigg)$

<Br>

#### 4.1.1. Example : Sample mean and Hoeffding's inequality



A simple example of McDiarmid's inequality in action is to see that it directly implies the Hoeffding bound. 


$$
f(X_1 , \cdots, X_n) = \frac{1}{n} \sum_{i=1}^n X_i \quad, for \quad a \leq X_i \leq b, ~\forall i
$$
since $\vert f(x_1, \cdots, x_{k-1}, x_k, x_{k+1}, \cdots , x_n) - f(x_1, \cdots, x_{k-1}, x_k', x_{k+1}, \cdots , x_n) \vert \leq \frac{(b-a)}{n}$,


$$
\mathbb{P} ( \vert \bar{X} - E[\bar {X}] \geq t) \leq 2 exp \bigg( -\frac{2t^2}{n \times \frac{(b-a)^2}{n^2}}\bigg) = 2 e^{\frac{2nt^2}{(b-a)^2}} 
$$
<br>

### 4.2. U-statistics

A perhaps more interesting example is that of <b> U-statistics</b>.



<b> Definition  (U-statistics)</b>  Let $X_1, \cdots, X_n$ be $i.i.d.$ random variables from some distribution $P$ supported on $\mathcal{X}$ . Consider a function $h$ that is symmetric in its arguments and satisfies $\underset{x_1, cdots, x_m \in \mathcal{X}}{sup}  \big\vert h(x_1, \cdots ,x_m ) \big\vert \leq K$ for some constant $K$. Let $\Sigma_{n,m}$ denote the summation taken over all subsets $1 \leq i_1< \cdots < i_m \leq n $ of $\{1,\cdots ,n\}$. Then


$$
U_n = \begin{pmatrix} n \\ m\end{pmatrix} ^{-1} \sum_{(n,m)} h(X_{i1}, \cdots , X_{im})
$$


is called a <b> U-statistics </b> with kernel $h$ of order $m$

<br>



#### 4.2.1. Example with order 2 and Bounded difference condition 

Suppose $m=2$ and $\vert h(X_i, X_j) \vert \leq b$, then,


$$
\vert U(x_1, \cdots, x_{k-1}, x_k, x_{k+1}, \cdots , x_n) - U(x_1, \cdots, x_{k-1}, x_k', x_{k+1}, \cdots , x_n) \vert  \leq \pmatrix{n \\2}^{-1} (n-1)(2b) = \frac{4b}{n}
$$


Then, by Mcdiarmid's inequality, 


$$
\mathbb{P}(\vert U(x_1, \cdots, x_{k-1}, x_k, x_{k+1}, \cdots , x_n) - U(x_1, \cdots, x_{k-1}, x_k', x_{k+1}, \cdots , x_n) \vert \geq t) \leq 2 exp(-nt^2 / 8b^2)
$$
<Br>

#### 4.2.2. Concentration Inequalities for U-statistics

<B> Theorem) </b> for the <b> U-statistics </b> of above definition, $U_n $ satisfies following concentration inequalities,
$$
\mathbb{P} (\vert U_n - \mathbb{E} [U_n] \geq t) \leq 2 \exp{ \left\{ - \frac{t^2 \lfloor{n/m \rfloor} } {2K^2}\right\}}
$$


<b> proof) </b>



Proof will follow two step, 



<b> Step 1) </b> Let $k = \lfloor \frac{n}{m} \rfloor$ , then 


$$
V_n =  \frac{1}{k} \{ h(X_1, \cdots, X_m) + ,\cdots, h(X_{(k-1)m+1}, \cdots, X_{km}) \}
$$


has concentration bound : 


$$
\mathbb{P} (\vert V_n - \mathbb{E} [V_n ] \geq t ) \leq 2 \exp \left\{ - \frac{t^2\lfloor \frac{n}{m} \rfloor}{2K^2} \right\}
$$


<b>pf)</b>  Notate $(X_{i1}, \cdots, X_{im}) = Y_i$,  then,



we can use notation of $V_n (Y_1, \cdots, Y_k) = \frac{1}{k} \{ h(Y_1) + \cdots + h(Y_k)  \} $



Here, one can check bounded difference condition : 


$$
\vert V_n (Y_1, \cdots , Y_i , \cdots , Y_k) - V_n (Y_1, \cdots , Y_i ' , \cdots , Y_k) \vert  = \vert \frac{1}{k} (h(Y_i) - h(Y_i ')\vert \leq \frac{2K}{k}
$$


Then, applying McDiarmid's theorem, 


$$
P( \vert V_n - E[V_n]\vert \geq t) \leq 2 \exp \bigg\{ - \frac{t^2k}{2K^2} \bigg\} = 2 \exp \bigg\{ - \frac{t^2\lfloor \frac{n}{m} \rfloor}{2K^2} \bigg\}
$$


Also, above inequality means that $V_n $ is sub-Gaussian with $\sigma^2 = \frac{K^2}{k}$.

<Br>

<B> Step 2) </b> Let's say $X_{\sigma} = (X_{\sigma_1 }, \cdots , X_{\sigma_n})$ to be permutation of $X$ and $S^n$ is possible group of such permutations. I will use of the fact that $V_n$ is invariant to permutation, which means that  


$$
\vert V_n (Y_{\sigma 1}, \cdots , Y_{\sigma i} , \cdots , Y_{\sigma k}) - V_n (Y_{\sigma 1}, \cdots , Y_{\sigma i} ' , \cdots , Y_{\sigma k}) \vert  = \vert \frac{1}{k} (h(Y_{\sigma i}) - h(Y_{\sigma i} ')\vert \leq \frac{2K}{k}
$$


is invariant to permutation. Therefore, permutated $V_{\sigma n}$ still follows same sub-Gaussian.



Then, 


$$
\sum_{\sigma \in S^n} V_n (X_\sigma) = m! (n-m)! \sum_{i_1 < \cdots < i_m} h(X_{i1}, \cdots, X_{im}) = n! U_n \\
\rightarrow U_n = \frac{1}{n!} \sum_{\sigma \in S^n} V_n (X_\sigma)
$$


To calculate concentration, 


$$
\begin{align}

\mathbb{P} [U_n - E[U_n] \geq t] & \leq e^{-\lambda t} \mathbb{E} [e^{\lambda (U_n - E[U_n])}] ,~~\longleftarrow \lambda>0 ,~~markov \\
& = e^{-\lambda t} \mathbb{E} [e^{\lambda  \frac{1}{n!} (\sum_{\sigma \in S_n }(V_n - E(V_n)))}] \\
& \leq e^{-\lambda t} \frac{1}{n!} \sum_{\sigma \in S_n} \mathbb{E} [e^{\lambda (V_n - E[V_n])}]\\ 
& \leq e^{-\lambda t} e^{\frac{\lambda^2 K^2} {2k}}, \quad \longleftarrow \quad def \,\, of \,\, SG \\
& =\exp \bigg\{ - \frac{t^2k}{2K^2} \bigg\}, ~~\longleftarrow with ~~\lambda = \frac{kt}{K^2 }\\&= \exp \bigg\{ - \frac{t^2\lfloor \frac{n}{m} \rfloor}{2K^2} \bigg\}

\end{align}
$$


Also, we can apply same procedure in the other way, which means that 


$$
\mathbb{P} (\vert U_n - \mathbb{E} [U_n] \geq t) \leq 2 \exp{ \bigg\{ - \frac{t^2 \lfloor{n/m \rfloor} } {2K^2}\bigg\}}
$$
<br>

### 4.3. Levy's Inequality



