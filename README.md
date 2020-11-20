# Expectation-Maximization Algorithm for Mixed Linear Regression Problem
Implementation code for the Expectation-Maximization (EM) algorithm for solving mixed linear regression problem under Gaussian and Laplacain additive noise regime.
# Abstract 
Mixed linear regression (MLR) model is among the most exemplary statistical tools for modeling non-linear distributions using a mixture of linear models. When the additive noise in MLR model is Gaussian, Expectation-Maximization (EM) algorithm is  a widely-used algorithm  for  maximum likelihood estimation of MLR parameters. However, when noise is non-Gaussian, the steps of EM algorithm may not have closed-form update rules, which makes EM algorithm computationally expensive. In this work, we study the maximum likelihood estimation of the parameters of MLR model when the additive noise has non-Gaussian distribution.  In particular, we consider the case that noise has Laplacian distribution and we first
show that unlike the the Gaussian case, the resulting sub-problems of EM algorithm in this case does not have closed-form update rule and requirrs sovling linear programmming in each iteration. To overcome this issue, we use the idea of parrarelization and use multiple cors to do the compuation (Shichun ... ). 
# Problem Set-up

MLR is a generalization of a simple linear regression model in which the data points $\{(y_i, \bx_{i}) \in \mathbb{R}^ {d+1}\}_{i = 1}^N $ is generated by a mixture of linear components, i.e., 

<a href="https://www.codecogs.com/eqnedit.php?latex=y_i&space;=&space;\langle&space;\boldsymbol{\beta}_{\alpha_i}^*,&space;\textbf{x}_i&space;\rangle&space;&plus;&space;\epsilon_{i},\;\;\;&space;\quad&space;\forall&space;i&space;\in&space;\{1,\cdots,&space;N\}," target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i&space;=&space;\langle&space;\boldsymbol{\beta}_{\alpha_i}^*,&space;\textbf{x}_i&space;\rangle&space;&plus;&space;\epsilon_{i},\;\;\;&space;\quad&space;\forall&space;i&space;\in&space;\{1,\cdots,&space;N\}," title="y_i = \langle \boldsymbol{\beta}_{\alpha_i}^*, \textbf{x}_i \rangle + \epsilon_{i},\;\;\; \quad \forall i \in \{1,\cdots, N\}," /></a>

where  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\beta}_{\alpha_i}^*" title="\boldsymbol{\beta}_{\alpha_i}^*" />  are the ground-truth regression parameters. <img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon_{i}" title="\epsilon_{i}" /> is the<img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /> additive noise with probability density function <img src="https://latex.codecogs.com/gif.latex?\inline&space;f_{\epsilon}(\cdot)" title="f_{\epsilon}(\cdot)" />  and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha_i&space;\in&space;\{1,\cdots,&space;K\}" title="\alpha_i \in \{1,\cdots, K\}" /> where  <img src="https://latex.codecogs.com/gif.latex?\inline&space;P(\alpha_i&space;=&space;k)&space;=&space;p_k" title="P(\alpha_i = k) = p_k" /> with <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sum_{k&space;=&space;1}^K&space;p_k=&space;1" title="\sum_{k = 1}^K p_k= 1" />. For simplicity of notation, we define <img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\beta}^*&space;=&space;[\boldsymbol{\beta}_1^*,\cdots,\boldsymbol{\beta}_K^*]" title="\boldsymbol{\beta}^* = [\boldsymbol{\beta}_1^*,\cdots,\boldsymbol{\beta}_K^*]" />.



In this work, we assume that <img src="https://latex.codecogs.com/gif.latex?\inline&space;p_k&space;=&space;\frac{1}{K},\;&space;\forall&space;k&space;\in&space;\{1,\cdots,&space;K\}" title="p_k = \frac{1}{K},\; \forall k \in \{1,\cdots, K\}" /> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\{\epsilon_i\}_{i&space;=&space;1}^N" title="\{\epsilon_i\}_{i = 1}^N" /> are independent and identically distributed with probability density function <img src="https://latex.codecogs.com/gif.latex?\inline&space;f_{\epsilon}(\cdot)" title="f_{\epsilon}(\cdot)" />that has Gaussian or Laplacian distribution, i.e.,

<img src="https://latex.codecogs.com/gif.latex?\inline&space;f_{\epsilon}(\epsilon)&space;=&space;\frac{1}{\sqrt{2\pi\sigma^2}}&space;e^{-\frac{\epsilon^2}{2\sigma^2}}~\textit{(in&space;Gaussian&space;scenario)}" title="f_{\epsilon}(\epsilon) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{\epsilon^2}{2\sigma^2}}~\textit{(in Gaussian scenario)}" />
<img src="https://latex.codecogs.com/gif.latex?\inline&space;f_{\epsilon}(\epsilon)&space;=&space;\frac{1}{2b}&space;e^{-\frac{|\epsilon|}{b}},&space;b&space;=&space;\frac{\sigma}{\sqrt{2}}~\textit{(in&space;Laplacian&space;scenario)}" title="f_{\epsilon}(\epsilon) = \frac{1}{2b} e^{-\frac{|\epsilon|}{b}}, b = \frac{\sigma}{\sqrt{2}}~\textit{(in Laplacian scenario)}" />

where <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma" title="\sigma" /> is the standard deviation of each distribution that is assumed to be known a priori. This limited choice of the additive noise is based on the fact that these two distributions cover wide range of applications such as medical image denoising [[1]](#1), video retrieval [[2]](#2) and clustering trajectories [[3]](#3).  



Our goal is inferring <img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\beta}^*" title="\boldsymbol{\beta}^*" /> given <img src="https://latex.codecogs.com/gif.latex?\inline&space;\{(y_i,&space;\textbf{x}_{i})\}_{i&space;=&space;1}^N" title="\{(y_i, \textbf{x}_{i})\}_{i = 1}^N" /> via Maximum likelihood estimator (MLE), which is the commonly used in practice [[4]](#4). Given the described model, the MLE <img src="https://latex.codecogs.com/gif.latex?\inline&space;\widehat{\boldsymbol{\beta}}" title="\widehat{\boldsymbol{\beta}}" /> can be computed by solving: 

<img src="https://latex.codecogs.com/gif.latex?\hat{\boldsymbol{\beta}}=&space;\arg\max_{\boldsymbol{\beta}}&space;\;&space;\log&space;\mathcal{P}(y_1,\ldots,&space;y_N|&space;\textbf{X},&space;\boldsymbol{\beta})" title="\hat{\boldsymbol{\beta}}= \arg\max_{\boldsymbol{\beta}} \; \log \mathcal{P}(y_1,\ldots, y_N| \textbf{X}, \boldsymbol{\beta})" />


Next, we will discuss how to solve this problem in both Gaussian and Laplacian cases using EM algorithm. 

# Expectation-Maximization Algorithm
EM algorithm is an iterative method that in each iteration finds a tight lower-bound for the objective function of the MLE problem and maximizes that lower-bound at that iteration [[5]](#5)[[6]](#6) . More precisely, the first step (E-step) involves updating the latent data labels  and the second step (M-step) includes updating the parameters. That is, the first step updates the probability of each data point belonging to different labels given the estimated coefficients, and the second step updates the coefficients given the label of all data. Let <img src="https://latex.codecogs.com/gif.latex?\inline&space;{\boldsymbol{\beta}}^{t}&space;=&space;({\boldsymbol{\beta}}_{1}^{t},\cdots,&space;{\boldsymbol{\beta}}^{t}_{K})" title="{\boldsymbol{\beta}}^{t} = ({\boldsymbol{\beta}}_{1}^{t},\cdots, {\boldsymbol{\beta}}^{t}_{K})" /> be the estimated regressors and <img src="https://latex.codecogs.com/gif.latex?\inline&space;{w}_{k,i}^{t}" title="{w}_{k,i}^{t}" /> be the probability that $<img src="https://latex.codecogs.com/gif.latex?\inline&space;i^{th}" title="i^{th}" /> data belongs to <img src="https://latex.codecogs.com/gif.latex?\inline&space;k^{th}" title="k^{th}" /> component at iteration <img src="https://latex.codecogs.com/gif.latex?\inline&space;t" title="t" />. Starting from the initial points <img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\theta}^0" title="\boldsymbol{\theta}^0" /> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;w&space;^{0}_{k,i}" title="w ^{0}_{k,i}" />, two major steps of the EM algorithm is as following,

E-step:


<img src="https://latex.codecogs.com/gif.latex?\inline&space;{w}_{k,i}^{t&plus;1}=&space;\frac{f_{\epsilon}(y_i-&space;\langle&space;\textbf{x}_{i},&space;\boldsymbol{\beta}_{k}^{t}\rangle)}{\sum\limits_{j&space;=&space;1}^{K}f_{\epsilon}(y_i-&space;\langle&space;\textbf{x}_{i},&space;\boldsymbol{\beta}^{t}_{j}\rangle)&space;},&space;\;&space;\forall&space;k,i," title="{w}_{k,i}^{t+1}= \frac{f_{\epsilon}(y_i- \langle \textbf{x}_{i}, \boldsymbol{\beta}_{k}^{t}\rangle)}{\sum\limits_{j = 1}^{K}f_{\epsilon}(y_i- \langle \textbf{x}_{i}, \boldsymbol{\beta}^{t}_{j}\rangle) }, \; \forall k,i," />



M-step:


<img src="https://latex.codecogs.com/gif.latex?\inline&space;{\boldsymbol{\beta}}^{t&plus;1}&space;=&space;\arg\min_{\boldsymbol{\beta}}&space;-&space;\sum_{i&space;=&space;1}^{N}\sum_{k&space;=&space;1}^{K}&space;{w}_{k,i}^{t&plus;1}&space;\log&space;\;&space;f_{\epsilon}(y_i&space;-&space;\langle&space;\boldsymbol{\beta}_k,&space;\textbf{x}_i&space;\rangle)" title="{\boldsymbol{\beta}}^{t+1} = \arg\min_{\boldsymbol{\beta}} - \sum_{i = 1}^{N}\sum_{k = 1}^{K} {w}_{k,i}^{t+1} \log \; f_{\epsilon}(y_i - \langle \boldsymbol{\beta}_k, \textbf{x}_i \rangle)" />

The problem in~\eqref{M-step} is separable with respect to <img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\beta}_k" title="\boldsymbol{\beta}_k" />'s. Thus, we can estimate  <img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\beta}_k^{t&plus;1}" title="\boldsymbol{\beta}_k^{t+1}" />'s in parallel by solving

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\beta}^{t&plus;1}_{k}=\arg\min_{\boldsymbol{\beta}_k}&space;-&space;\sum_{i&space;=&space;1}^{N}&space;{w}_{k,i}^{t&plus;1}&space;\log&space;\;&space;f_{\epsilon}(y_i&space;-&space;\langle&space;\boldsymbol{\beta}_k,&space;\textbf{x}_i&space;\rangle),&space;\forall&space;k." title="\boldsymbol{\beta}^{t+1}_{k}=\arg\min_{\boldsymbol{\beta}_k} - \sum_{i = 1}^{N} {w}_{k,i}^{t+1} \log \; f_{\epsilon}(y_i - \langle \boldsymbol{\beta}_k, \textbf{x}_i \rangle), \forall k." />

Let us discuss this optimization problem in two cases of Gaussian and Laplacian noise scenarios:

# Additive Gaussian noise
When the additive noise has Gaussian distribution, problem~\eqref{M-step-seperate} is equivalent to

<img src="https://latex.codecogs.com/gif.latex?\inline&space;\boldsymbol{\beta}^{t&plus;1}_k&space;=&space;\arg\min_{\boldsymbol{\beta}_k}&space;\sum_{i&space;=&space;1}^{N}&space;{w}_{k,i}^{t&plus;1}&space;(y_i&space;-&space;\langle&space;\boldsymbol{\beta}_k,&space;\textbf{x}_i&space;\rangle)^2,&space;\quad&space;\forall&space;k." title="\boldsymbol{\beta}^{t+1}_k = \arg\min_{\boldsymbol{\beta}_k} \sum_{i = 1}^{N} {w}_{k,i}^{t+1} (y_i - \langle \boldsymbol{\beta}_k, \textbf{x}_i \rangle)^2, \quad \forall k." />


It can be easily shown that this problem has the closed-form solution of the form  
<img src="https://latex.codecogs.com/gif.latex?\inline&space;{\boldsymbol{\beta}}_k^{t&plus;1}&space;=&space;(\sum_{i&space;=&space;1}^{N}&space;{w}_{k,i}^{t&plus;1}&space;\textbf{x}_i&space;\textbf{x}_i^{T})^{-1}&space;\sum_{&space;i&space;=&space;1}^{N}&space;{w}_{k,i}^{t&plus;1}&space;y_i&space;\textbf{x}_i,&space;\;\;\;\forall&space;k." title="{\boldsymbol{\beta}}_k^{t+1} = (\sum_{i = 1}^{N} {w}_{k,i}^{t+1} \textbf{x}_i \textbf{x}_i^{T})^{-1} \sum_{ i = 1}^{N} {w}_{k,i}^{t+1} y_i \textbf{x}_i, \;\;\;\forall k." />


# Additive Laplacian noise 
For the Laplacian case, the problem in~\eqref{M-step-seperate} is equivalent to 

<img src="https://latex.codecogs.com/gif.latex?\inline&space;{\boldsymbol{\beta}}^{t&plus;1}_{k}&space;=&space;\arg\min_{\boldsymbol{\beta}_k}&space;\;&space;\sum_{i&space;=&space;1}^{N}{w}_{k,i}^t&space;\;\;&space;|y_i&space;-&space;\langle&space;\boldsymbol{\beta}_k,&space;\textbf{x}_i&space;\rangle|,\quad&space;\forall&space;k." title="{\boldsymbol{\beta}}^{t+1}_{k} = \arg\min_{\boldsymbol{\beta}_k} \; \sum_{i = 1}^{N}{w}_{k,i}^t \;\; |y_i - \langle \boldsymbol{\beta}_k, \textbf{x}_i \rangle|,\quad \forall k." />

Despite convexity of this problem,  this optimization problem is non-smooth. Thus, we use sub-gradientfor solving it.

# Summary of the EM algorithm 
The idea behind the proposed algorithm is that in each iteration, the maximization problem is solved to a good accuracy. This gives us an estimate of the gradient of the minimization problem. This gradient is later used for solving the outer minimization problem.
<p align="center">
  <img width="380" height="300" src="https://github.com/babakbarazandeh/Mixed_Linear_Regression/blob/main/EM.png">
</p>
 
# Reuslts 

## References
<a id="1">[1]</a> 
B. Klein, G. Lev, G. Sadeh, and L. Wolf, “Fisher
vectors derived from hybrid gaussian-laplacian mixture models for image annotation,” arXiv preprint
arXiv:1411.7399, 2014.

<a id="2">[2]</a> 
T. Amin, M. Zeytinoglu, and L. Guan, “Application of
laplacian mixture model to image and video retrieval,”
IEEE Transactions on Multimedia, vol. 9, no. 7, pp.
1416–1429, 2007.

<a id="3">[3]</a> 
S. Gaffney and P. Smyth, “Trajectory clustering with
mixtures of regression models,” in Proceedings of
the fifth ACM SIGKDD international conference on
Knowledge discovery and data mining, 1999, pp. 63–72.

<a id="4">[4]</a> 
 K. Zhong, P. Jain, and I. S. Dhillon, “Mixed linear
regression with multiple components,” in Advances
in neural information processing systems, 2016, pp.
2190–2198.
<a id="5">[5]</a> 
M. Razaviyayn, M. Hong, and Z.-Q. Luo, “A unified
convergence analysis of block successive minimization
methods for nonsmooth optimization,” SIAM Journal
on Optimization, vol. 23, no. 2, pp. 1126–1153, 2013.
<a id="6">[6]</a> 
 A. P. Dempster, N. M. Laird, and D. B. Rubin,
“Maximum likelihood from incomplete data via the em
algorithm,” Journal of the Royal Statistical Society:
Series B (Methodological), vol. 39, no. 1, pp. 1–22,
1977
# Getting started
Run Main.m

