---
layout: post
title: EM Algorithm in Matlab (Auxillary Information and Code)
category: Machine Learning
date: 19-Sep-2020
---
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML">
</script>

This post is just some more additional information from my [previous one]((2020-09-14-EM-Algorithm.md)) (read that one first). This covers some debugging tips and other things to be aware of.

**Underflow**

Firstly, in MATLAB at least, you may run into some underflow problems like I did (e.g. `badly conditioned matrix` or some division by zero error). One way to allieviate this is to use something called the log-sum-exp trick. I have a post [here](2020-09-14-logsumtrick.md) that explains the trick in more detail. Basically, we just perform all our calculations on the log probability and then raise it to $e$ after we're done. 

**Convergence**

Sometimes, just for sanity's sake, we want to try plot the convergence:
[![convergence]({{site.url}}\pics\convergence.JPG)]
```Matlab
For each iteration t {
    A = max(gam,[],1);  %max for every point
    T = A + log(sum(exp(gam - A))); %T is the the sum along every column
    llh(t) = sum(T)/n; % loglikelihood
}
```
Note that the above also uses the log-sum-exp trick. We can simply sum over all the $\gamma_k$ instead.