---
layout: post
title: Why (and when) is Euclidean Distance bad? 
category: Bible
date: 22-Sep-2020
---

 Euclidean distance is not necessarily a good metric for higher dimensions (especially for $200-400$ dimensions), as the probability of finding a point no longer follows normal intuition as with  1 or 2D. 
 
 Consider a gaussian distributed dataset: rather than the probability being distributed acoording to a gaussian, most of the points lie one the surface of a n-sphere i.e. on the 'skin' of a ball instead of the insides. The implication of this is that data becomes very sparse and the distance ratios (and thus the similarities) all approach one.
$$
d_{\text {cosine}} x, y=\frac{\langle x, y\rangle}{|x||y|}
$$
If we consider cosine distance, then we instead are measuring the angles between points instead of magnitude. That is, the magnitude does not matter anymore, only the angles from a mean. In this regard, we can still measure the similarity between two samples based of the angle in high dimensions.

## Example:
In speaker verification systems, it is common to represent speech utterances as vectors which are typically $200-400$ dimensional vectors that nominally follow a standard normal distribution (mean is $\mathbf{0}$ and covariance is $\mathbf{I}$ ).

In speech/word utterances, there will typically have multiple utterances that occur a lot more than the other, which creates variability in the magnitudes. For example, if we want to capture the similarity between two sources that include the utterance 'ah' and 'o', then it should not matter how many times these utterances occur. Using oosine distance allows more resilience to variation of the frequency of the utterances.