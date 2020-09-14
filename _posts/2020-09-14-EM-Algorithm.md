---
layout: post
title: EM Algorithm in Matlab
category: Machine Learning
date: 14-Sep-2020
---
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML">
</script>

The EM Algorithm is a unsupervised clustering method that tries to fit gaussian mixture models (GMM) to data. The important formulas are below, split into two steps: the Expectation step and the Maximization step. Proof can probably be found somewhere not here.

First, we define our gaussian as so: 

**Gaussian Distribution**

$$
\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \mathbf{\Sigma})=\frac{1}{\sqrt{(2 \pi)^{d}|\Sigma|}} e^{\left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1}(x-\mu)\right)}
$$

**E-Step**

$$
\gamma_{n k}=\frac{\pi_{k} \mathcal{N}\left(x_{n} \mid \mu_{k}^{\text {old }}, \Sigma_{k}^{\text {old }}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(x_{n} \mid \mu_{j}^{\text {old }}, \Sigma_{j}^{\text {old }}\right)}
$$

**M-Step**

$$
\mu_{k}^{n e w}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k} x_{n} \\
\Sigma_{k}^{n e w}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k}\left(x_{n}-\mu_{k}^{{new}}\right)\left(x_{n}-\mu_{k}^{{new}}\right)^{T} \\
\pi_{k}^{n e w}=\frac{N_{k}}{N}
$$