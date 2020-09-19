---
layout: post
title: EM Algorithm in Matlab
category: Machine Learning
date: 14-Sep-2020
---
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML">
</script>

The EM Algorithm is a unsupervised clustering method that tries to fit gaussian mixture models (GMM) to data. The important formulas are shown in this post, with the algorithm mainly iterating between two steps: the Expectation step and the Maximization step. I wont go into the proof, but I will try to explain what each part of the formulae does. If you want to know more, check out the links at the bottom - they expand on a great deal more than I do.

## When to use EM?
Say we have some unlabelled data, and we want to fit some parametric model to it. Usually, we want to find some **maximum likelihood** estimation for the data. However, we may have some incomplete/unobserved variables which may screw things up. Also see [this](https://stats.stackexchange.com/questions/326343/why-use-em-algorithm-instead-of-just-plain-old-ml-for-mixture-model) answer for more information.

As for how the algorithm actually works, this nice gif from [Wikipedia](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) quite nicely shows the iterative steps:
[![EM Algorithm](https://upload.wikimedia.org/wikipedia/commons/6/69/EM_Clustering_of_Old_Faithful_data.gif)](https://upload.wikimedia.org/wikipedia/commons/6/69/EM_Clustering_of_Old_Faithful_data.gif)

First, we define our gaussian as so: 

**Gaussian pdf**

$$
\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \mathbf{\Sigma})=\frac{1}{\sqrt{(2 \pi)^{d}|\Sigma|}} e^{\left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)}
$$

so given a $d$-dimension mean $\mu$ and $d\times d$ covariance matrix $\Sigma$, we have the gaussian distribution.

```Matlab
function P = my_norm(x, mu, Sigma)
    %For E-step. calculates P(xi|b) according to gaussian/normal dist
    % x is a vector in the x1, x2... (data)
    % mu is the mean (estimate). should be the same lenge as x
    % Sigma is the covariance matrix
    d = length(mu);
    P = 1/((2*pi)^(d/2)*sqrt(det(Sigma)))*exp(-0.5*(x-mu)/Sigma*(x-mu)');
end
```

**E-Step**

The first step is the Expectation step. 

$$
\gamma_{n k}=\frac{\pi_{k} \mathcal{N}\left(x_{n} \mid \mu_{k}^{\text {old }}, \Sigma_{k}^{\text {old }}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(x_{n} \mid \mu_{j}^{\text {old }}, \Sigma_{j}^{\text {old }}\right)}
$$

Here $\gamma_{nk}$ can be thought of as the likelihood that a point $x_n$ corresponds to a cluster $k$. $\pi_k$ is the weight corresponding to cluster $k$.

```Matlab
for j = 1:k
    for i = 1:n
        gam(j,i) = my_norm(X(i,:),mu(j,:), sig(:,:,j));
    end
    gam(j,:) = pis(j)*gam(j,:);
end
gam = gam./sum(gam);
```
**M-Step**

After the E-step, we update the means, covariances, and weights:

$$
\mu_{k}^{n e w}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k} x_{n} 
$$

$$
\Sigma_{k}^{new}=\frac{1}{N_{k}} \sum_{n=1}^{N}\gamma_{nk}\left(x_{n}-\mu_{k}^{new}\right)\left(x_{n}-\mu_{k}^{new}\right)^{T}
$$

$$
\pi_{k}^{n e w}=\frac{N_{k}}{N}
$$

$N_k$ is just the sum of all $\gamma_n$ for a cluster $k$. 

```Matlab
%update mu
mu = zeros(k,p);
for j = 1:k
    mu(j,:) = gam(j,:)*X;
    mu(j,:) = mu(j,:)./sum(gam(j,:));
end

%update covariance. Note we use updated mu
sig = zeros(p,p,k);
for j = 1:k
    %numerator
    for i = 1:n
        sig(:,:,j) = sig(:,:,j) + gam(j,i)*(X(i,:) - mu(j,:))'*(X(i,:) - mu(j,:));
    end
    sig(:,:,j) = sig(:,:,j)./sum(gam(j,:)); 
end

pis = zeros(k,1);
for j = 1:k
    pis(j) = sum(gam(j, :));
end
pis = pis./n;
```

Finally we can simply iterate E-step, and then M-step until it converges. Some things to note however, is that you may run into underflow problems if the probablity becomes too small. See [my logsumtrick post](2020-09-14-logsumtrick.md) for a way to alleviate it. You can also check convergence by plotting the log likelihood (see the next post [here](2020-09-19-EM-Auxillary.md))



## Resources
- (2D blog Example) [https://medium.com/@jonathan_hui/machine-learning-expectation-maximization-algorithm-em-2e954cb76959](https://medium.com/@jonathan_hui/machine-learning-expectation-maximization-algorithm-em-2e954cb76959)
- Code

**Videos**
- StatQuest: Maximum Likelihood (for proof) [https://www.youtube.com/watch?v=XepXtl9YKwc](https://www.youtube.com/watch?v=XepXtl9YKwc) 

- EM Algorithm (should watch for intuition) [https://www.youtube.com/watch?v=REypj2sy_5U&ab_channel=VictorLavrenko](https://www.youtube.com/watch?v=REypj2sy_5U&ab_channel=VictorLavrenko)

**Python Code (quite long)**
- [https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html](https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html)
- [https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php](https://www.python-course.eu/expectation_maximization_and_gaussian_mixture_models.php)