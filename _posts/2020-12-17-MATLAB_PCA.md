---
layout: post
title: How to use Matlab's PCA
category: Machine Learning
date: 20-Sep-2020
---

<div class="content">

## Contents

<div>

- [Contents](#contents)
- [How to use Matlab's PCA](#how-to-use-matlabs-pca)
- [Run PCA](#run-pca)
- [Project the data into PCA space](#project-the-data-into-pca-space)
- [Show variance](#show-variance)
- [Correlation](#correlation)
- [Dimensionality reduction](#dimensionality-reduction)
- [Comparison](#comparison)

</div>

## How to use Matlab's PCA

The documentation for Matlab's PCA made no sense, so here is a worked example on how to use it and what the variables actually mean. Derivation may be covered at a later date. The short version is that PCA computes principal components (eigenvectors that maximize variance), then uses them to perform a change of basis on the data. Sometimes it only uses the first few principal components and ignores the rest. For this example, we use the fisher iris classification dataset.

<pre class="codeinput">load <span class="string">fisheriris.mat</span>
X = meas;
X = X - mean(X); <span class="comment">%dont really have to do this, since matlab's pca() does it anyway</span>
</pre>

## Run PCA

<pre class="codeinput"><span class="comment">%Matlab's method</span>
<span class="comment">% returns...</span>
<span class="comment">%   coeff : Matrix W that contains the principal component</span>
<span class="comment">%   coefficients/vectors (the eigenvectors). score : principal component</span>
<span class="comment">%   scores. This is essentially the transformed data T in PCA space (found</span>
<span class="comment">%   by multiplying X*W). latent: principal component variances. Shows which</span>
<span class="comment">%   components are usefull</span>
[coeff,score,latent] = pca(X);

<span class="comment">% Derivation</span>
<span class="comment">% Calculate eigenvalues and eigenvectors of the covariance matrix</span>
<span class="comment">%   V : eigenvectors</span>
<span class="comment">%   D : eigenvalues</span>
covarianceMatrix = cov(X);
[V,D] = eig(covarianceMatrix);

<span class="comment">% In principle, V is the same as coeff</span>
<span class="comment">% (Note that the columns are not necessarily in the same *order*,</span>
<span class="comment">%  and they might be *lightly different from each other</span>
<span class="comment">%  due to floating-point error.)</span>
</pre>

## Project the data into PCA space

<pre class="codeinput"><span class="comment">%In principle, X_new is the same as scores</span>
X_new = X*coeff;

<span class="comment">% plot transformed PCA space.</span>
<span class="comment">% The score and the transformed data are the same, save that X_new is not</span>
<span class="comment">% scaled/normalized (need to subtract mean first)</span>
X_new = X_new - mean(X_new);

figure()
plot(score(:,1),score(:,2),<span class="string">'+'</span>)
hold <span class="string">on</span>
plot(X_new(:,1),X_new(:,2), <span class="string">'o'</span>)
legend(<span class="string">'matlab'</span>, <span class="string">'X\_new'</span>, <span class="string">'Location'</span>, <span class="string">'best'</span>)
xlabel(<span class="string">'1st Principal Component'</span>)
ylabel(<span class="string">'2nd Principal Component'</span>)
title(<span class="string">'Projected Space'</span>)
</pre>

![]({{site.url}}\\pics\pca_test_01.png)

## Show variance

A scree plot can be used to show contributions from each principle component, The y axis depicts the amount of variance Evidently, the first principle component (PC) is the most statistically significant

The plot also shows 3 methods of generating variance: 1) calculating variance from the new projected space X_new, 2) using matlab's pca(), 3) calculating the eigenvalue from eig()

<pre class="codeinput">figure,
hold <span class="string">on</span>
plot(var(X_new)', <span class="string">'x'</span>)
plot(latent,<span class="string">'o'</span>)
plot(sort(diag(D),<span class="string">'descend'</span>), <span class="string">'-.'</span>)
legend(<span class="string">'var(X_new)'</span>, <span class="string">'latent (matlab)'</span>, <span class="string">'manual eigenvalue'</span>)
title(<span class="string">'scree plot'</span>)
ylabel(<span class="string">'variance'</span>)
xlabel(<span class="string">'Component number'</span>)
</pre>

![]({{site.url}}\\pics\pca_test_02.png)

## Correlation

Original shows that multiple features are correlated. We want to remove that by projecting pca


<pre class="codeinput">figure,
corrplot(X)
title(<span class="string">'original'</span>)
figure,
corrplot(X_new)
title(<span class="string">'new'</span>)
</pre>

![]({{site.url}}\\pics\pca_test_03.png) ![]({{site.url}}\\pics\pca_test_04.png)

## Dimensionality reduction

If we want to perform some dimensionality reduction (DR), we can use PCA to retain as much variance as possible = the most informative features are found with high variances

Practically we can just throw out coef column like so

<pre class="codeinput">X_new_2d = X*coeff(:,1:2); <span class="comment">%keeping the first two columns => reduce to 2D</span>
</pre>

## Comparison

Plot a comparison between first two features and these features projected onto PCA space. We can see that the variance is maximized with the new (red x) points.

<pre class="codeinput">figure
hold <span class="string">on</span>
scatter(X(:,1), X(:,2), <span class="string">'x'</span>)
scatter(X_new(:,1), X_new(:,2), <span class="string">'o'</span>)
title(<span class="string">'feature 1 vs 2'</span>)
legend(<span class="string">'old'</span>, <span class="string">'new'</span>)
xlabel(<span class="string">'dim 1'</span>)
ylabel(<span class="string">'dim 2'</span>)
</pre>

![]({{site.url}}\\pics\pca_test_05.png)

<pre class="codeinput"><span class="comment">%(optional) check that coefficients are orthonormal</span>
coeff*coeff';
</pre>

[Published with MATLAB® R2019b](https://www.mathworks.com/products/matlab/)  

</div>