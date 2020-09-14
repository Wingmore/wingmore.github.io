---
layout: page
title: Avoiding Underflow with the log-sum-exp trick 
category: Machine Learning
date: 14-Sep-2020
---

The logsumtrick is a very com

Consider the following equation (used in the Expectation Maximization (EM) Algorithm)

$$\gamma_{n k}=\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}^{\text {old }}, \mathbf{\Sigma}_{k}^{\text {old }}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}^{\text {old }}, \mathbf{\Sigma}_{j}^{\text {vid }}\right)}$$

If we run the code straight up using the formulas described, then it is very likely that we run into an underflow problem as the probabilities may be very, very small. To prevent this issue, we can perform all numerical operations on the log probability, as this is much more practical for computation. In log space, multiplication becomes addition, and division becomes subtraction. This drastically improves speed, numerical stability, and simplicity while avoiding the risk of extremely large numbers (overflow) or extremely small numbers (underflow). 

So to compute γ_nk, we can take the (natural) log of the RHS, and then raise it to e after all operations. i.e. $γ_nk=e^(log⁡(γ_nk))$. Taking the log of the RHS  :

$$
\log \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}^{\text {old }}, \mathbf{\Sigma}_{k}^{\text {obd }}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}^{\text {old }}, \mathbf{\Sigma}_{j}^{\text {old }}\right)}=\log (N u m)-\log (D e n)
$$
For the numerator, we have
$$
\begin{aligned}
\log (N u m) &=\log \pi_{k}+\log \mathcal{N}\left(\boldsymbol{x}_{\boldsymbol{n}} \mid \boldsymbol{\mu}_{\boldsymbol{k}}^{\text {old }}, \mathbf{\Sigma}_{\boldsymbol{k}}^{\text {old }}\right) \\
&\left.=\log \pi_{k}+\log \frac{1}{\sqrt{(2 \pi)^{d}|\Sigma|}} e^{\left(-\frac{1}{2}(x-\mu)^{7} \Sigma^{-1}(x-\mu)\right.}\right) \\
&=\log \pi_{k}-\frac{1}{2}\left[\log (2 \pi)+\log (|\Sigma|)+(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right]
\end{aligned}
$$
