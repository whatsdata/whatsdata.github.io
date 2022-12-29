---
layout : single
title : '[Analysis] Uniform convergence'
categories : 'Mathematics'
sidebar_main : true
tag : ['stat', 'Analysis']
date : 2022-12-19
last_modified_at : 2022-12-19
---




- Pointwise Convergence

<Br>

Let $(N, d)$ be metric space. For  sequence of functions 


$$
\begin{align}
&f_n : A \longrightarrow N, \quad n =1, \cdots , n \\
&f\,\,\, : A \longrightarrow N
\end{align}
$$


The sequence of functions are said to <b> converge pointwise </b> to the function $f$, if <i> for each $x \in A$, $f_n (x) \longrightarrow f(x)$,  </i>

i.e. <i> for each $x\in A$ and $\epsilon >0$, $\exists L = L(x, \epsilon)$  such that $d(f_n (x), (x)) \leq \epsilon , \quad \forall n \geq L$  </i>



<Br>

- Uniform Convergence

<Br>

Let $(N, d)$ be metric space. For  sequence of functions 


$$
\begin{align}
&f_n : A \longrightarrow N, \quad n =1, \cdots , n \\
&f\,\,\, : A \longrightarrow N
\end{align}
$$


The sequence of functions are said to <b> converge pointwise </b> to the function $f$, if <i> for each $x \in A$, $f_n (x) \longrightarrow f(x)$,  </i>



i.e. <i> for each $x\in A$ and $\epsilon >0$, $\exists L = L( \epsilon)$  such  that  $d(f_n (x), (x)) \leq \epsilon , \quad \forall n \geq L$  </i>



which is same with


$$
\underset{x \in A} {sup} \quad d(f_n (x) , (x)) < \epsilon, \quad \forall n \geq L
$$

<br>



- Some relations

  

<b>1. The derivatives of a pointwise convergent sequence of functions do not have to converge. </b>

<br>

consider $X = \mathbb{R} $  and  $f_n (x) = \frac{1}{n} sin(n^2 x)$ . Then,




$$
\underset{n \rightarrow \infty}{lim} f_n (x) = 0
$$
intwise limit function is $f(x)=0$; the sequence of functions converges to 0. What about the derivatives of the sequence?


$$
f_n '(x) = n cos (n^2 x)
$$


and for most $x \in \mathbb{R}$, above derivative is unbounded, which means that it does not converge. 





<br>

<b> 2. The integrals of a pointwise convergent sequence of functions do not have to converge. </b>

<br>

Consider $X= [0,1],$ and  $f_n (x) = \frac{2 n^2 x}{(1+n^2 x^2)^2}.$ Then,


$$
lim_{n \rightarrow =\infty} f_n (x) = 0
$$


However, the integrals are




$$
\int^1_0 \frac{2 n^2 x dx}{(1+n^2 x^2)^2} \overset{u = 1+ n^2 x^2}{=} \int ^{1+n^2}_{1 }\frac{du}{u^2} = 1 - \frac{1}{1+n^2}
$$


Therfore, even thought $lim_{n \rightarrow =\infty} f_n (x) = 0$ for all $x \in X$, the intergral is 1 as $n \rightarrow \infty$



<br>

<b> 3. The limit of a pointwise convergent sequence of continuous functions does not have to be contuniuous</b>

<Br>

$A = [ 0, 1]$  and  $f_n(x) = x^n $. Then, 


$$
\underset{n \rightarrow \infty}{f_n (x)} = f(x) = \cases{0 \quad (0 \leq x 
<1)\\ 1 \quad (x=1)}
$$
It satisfies pointwise convegence, but limit is not continuous

<Br>

<b> 4. The uniform convergence implies pointwise convergence, but not the other way around. </b> 



Same example with above one.



If $f_n(x)$ converges uniformly, then the limit function must be $f(x) =0$ for $x \in [0,1)$ and $f(1) = 1$. Uniform convergence implies that for any $\epsilon >0 $ there is  $N_\epsilon \in \mathbb{N}$ such that $\vert x^n - f(x)\vert$ for all $n \geq N_\epsilon$ . Then, consier $\epsilon = \frac{1}{2}$. Then, there is $N $ such that for all $n \geq N$, $\vert x^n - f(x) \vert < \frac{1}{2}$. If we choose $n=N$ and $x = (\frac{3}{4})^N$, $f(x) = 0 $ and thus


$$
\vert f_N (x) - f(x) \vert = \frac{3}{4}
$$


contradicting our assumption.