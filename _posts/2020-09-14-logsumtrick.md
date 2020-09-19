---
layout: post
title: Avoiding Underflow with the log-sum-exp trick (EM Algorithm)
category: Machine Learning
date: 14-Sep-2020
---
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML">
</script>

*Edit 19-Sep-20: Added Code*

The logsumexp tick is not explicitly taught, but is incredibly useful in ML applications. When dealing with probabilities, the output may end up being incredibly small (close to zero), leading to **underflow**. Thus it is often useful to perform calculations on [log probabilities](https://en.wikipedia.org/wiki/Log_probability).

Consider the following equation (from an [EM Algorithm](2020-09-14-EM-Algorithm.md))

$$
\gamma_{n k}=\frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}^{\text {old }}, \mathbf{\Sigma}_{k}^{\text {old }}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}^{\text {old }}, \mathbf{\Sigma}_{j}^{\text {vid }}\right)}
$$

If we run the code straight up using the formulas described, then it is very likely that we run into an underflow problem as the probabilities may be very, very small. To prevent this issue, we can perform all numerical operations on the **log probability**, so multiplication becomes addition, and division becomes subtraction. Practically, this drastically improves speed, numerical stability, and simplicity while avoiding the risk of extremely large numbers (overflow) or extremely small numbers (underflow). 

So to compute γ_nk, we can take the (natural) log of the RHS, and then raise it to e after all operations. i.e. \\(γ_{nk}=e^{log⁡(γ_{nk})}\\). Taking the log of the RHS  :


$$
\log \frac{\pi_{k} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{k}^{\text {old }}, \mathbf{\Sigma}_{k}^{\text {obd }}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(\boldsymbol{x}_{n} \mid \boldsymbol{\mu}_{j}^{\text {old }}, \mathbf{\Sigma}_{j}^{\text {old }}\right)}=\log (N u m)-\log (D e n)
$$

For the numerator, we have

$$\begin{aligned}
\log (N u m) &=\log \pi_{k}+\log \mathcal{N}\left(\boldsymbol{x}_{\boldsymbol{n}} \mid \boldsymbol{\mu}_{\boldsymbol{k}}^{\text {old }}, \mathbf{\Sigma}_{\boldsymbol{k}}^{\text {old }}\right) \\
&=\log \pi_{k}+\log \frac{1}{\sqrt{(2 \pi)^{d}|\Sigma|}} e^{\left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)} \\
&=\log \pi_{k}-\frac{1}{2}\left[\log (2 \pi)+\log (|\Sigma|)+(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right]
\end{aligned}$$

For the denominator, we cannot simply take the log of the sums since \\(\mathcal{N}\left(\boldsymbol(x_n \mid \mu_k^{old}, \Sigma_k^{old}\right)\\)
 may be very small and so $\log (\operatorname{sum}(\exp (-))$ leads to underflow. We can circumvent this problem by using the **log-sum-exp trick.** For convenience, we write \\(\mathcal{N}\left(\boldsymbol(x_n \mid \mu_k^{old}, \Sigma_k^{old}\right)\\). The denominator becomes:

$$
\log (D e n)=\log \left(\sum \pi_{k} P\right)=\log \left(\sum e^{\log \left(\pi_{k} P\right)}\right)=A+\log \left(\sum e^{\log \left(\pi_{k} P\right)-A}\right)
$$

**Explanation**: we first convert the denominator into a log-sum-exp form:\\(log \left(\sum e^{x}\right)\\), and then we apply the **log-sum-exp trick**, which lets us write \\(\log \left(\sum e^{a}\right)\\) as \\(A+\log \left(\sum e^{a-A}\right)\\). By setting the
variable $A$ as the max value in the sequence $a,$ we can prevent underflow since if all the numbers
in $x$ are very large negatively $($ e.g $a=[-222,-255]),$ then subtracting the maximum will bring it back close to $0(a-A=[0,3])$ which lets us perform log and exponential operations. If we do not subtract by A first, then $e^{a} \rightarrow 0,$ and $\log e^{a} \rightarrow inf$  instead of $a$.

## Code for the E-Step
This is the log calculation for the E-step. The original code is found in [this](2020-09-14-EM-Algorithm.md) post for comparison (does not use any log).

```Matlab
gam = zeros(k,n);   %the likelihood that a point belongs to k
%find log of numerator - ln(pi*P(x|b))
for j = 1:k
    for i = 1:n
        %log of gaussian
        gam(j,i) = my_lognorm(X(i,:),mu(j,:), sig(:,:,j));
        %multiply pis*P(x|b) ->> same as addition in log space
        gam(j,i) = gam(j,i) + log(pis(j));
    end
end
%find log of denominator sum - ln(SUM (pi P))
%do log sum trick here. See report for more info
A = max(gam,[],1);  %max for every point
T = A + log(sum(exp(gam - A))); %T is the the sum along every column
gam = exp(gam - T);
```

The my_lognorm is just this:
```Matlab
function P = my_lognorm(x, mu, Sigma)
    %For E-step. calculates P(xi|b) according to gaussian/normal dist
    % x is a vector in the x1, x2... (data)
    % mu is the mean (estimate). should be the same length as x
    % Sigma is the covariance matrix
    d = length(mu);
    P = -1/2*(d*log(2*pi)+log(det(Sigma)) + (x-mu)/Sigma*(x-mu)');
end
```

# Resources
- (example) [https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes](https://stats.stackexchange.com/questions/105602/example-of-how-the-log-sum-exp-trick-works-in-naive-bayes)
- (proof) [https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/](https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/)

