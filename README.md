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
\begin{align*}
&f_{\epsilon}(\epsilon) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{\epsilon^2}{2\sigma^2}}~\textit{(in Gaussian scenario)}, 
\\
&f_{\epsilon}(\epsilon) = \frac{1}{2b} e^{-\frac{|\epsilon|}{b}}, b = \frac{\sigma}{\sqrt{2}}~\textit{(in Laplacian scenario)},
\end{align*}
where $\sigma$ is the standard deviation of each distribution that is assumed to be known a priori. This limited choice of the additive noise is based on the fact that these two distributions cover wide range of applications such as medical image denoising~\cite{bhowmick2006laplace,klein2014fisher}, video retrieval~\cite{amin2007application} and clustering trajectories~\cite{gaffney1999trajectory}.  



Our goal is inferring $\boldsymbol{\beta}^*$ given $\{(y_i, \x_{i})\}_{i = 1}^N $ via Maximum likelihood estimator (MLE), which is the commonly used in practice~\cite{zhong2016mixed}. Given the described model, the MLE $\widehat{\boldsymbol{\beta}}$ can be computed by solving: 

\begin{align}\label{eg:Likelihood}
\nonumber
\hat{\boldsymbol{\beta}}= \arg\max_{\boldsymbol{\beta}} \; & \log \mathcal{P}(y_1,\ldots, y_N| \X, \boldsymbol{\beta})
\\ \nonumber
= \arg\max_{\boldsymbol{\beta}} \; & \sum_{i = 1}^N \log \mathcal{P}(y_i| \bx_i,\boldsymbol{\beta})
\\ 
= \arg\max_{\boldsymbol{\beta}} \; &  \sum_{i = 1}^N   \log \left( \sum_{k = 1}^{K} p_{k} f_{\epsilon}(y_i- \langle \textbf{x}_{i}, {\boldsymbol{\beta}}_{k}\rangle)\right) 
\end{align}

Next, we will discuss how to solve this problem. 
# Summary of the EM algorithm 
The idea behind the proposed algorithm is that in each iteration, the maximization problem is solved to a good accuracy. This gives us an estimate of the gradient of the minimization problem. This gradient is later used for solving the outer minimization problem.
<p align="center">
  <img width="380" height="300" src="https://github.com/babakbarazandeh/Multi-step-Accelerated-Proximal-Gradient-Descent-Ascent/blob/master/Algorithm.png">
</p>
 
# Reuslts 
We evaluate the performance of the proposed algorithm for the problem of attacking the LASSO estimator. In other words, our goal is to find a small perturbation of the observation matrix that worsens the performance of the LASSO estimator in the training set.
Following figure shows the performance of the Proposed Algorithm (PA) compared to Subgradient Descent-Ascent(SDA), and Proximal Descent-Ascent algorithm (PDA).  
To have a fair comparison, all of the studied algorithms have been initialized at the same random points.
<p align="center">
  <img width="500" height="200" src="https://github.com/babakbarazandeh/Multi-step-Accelerated-Proximal-Gradient-Descent-Ascent/blob/master/Result.png">
</p> <br/>

The above figure might not  be  a fair comparison  since each step of the proposed algorithm is computationally more expensive than the two benchmark methods.  To have a better comparison,we evaluate the performance of the algorithms in terms of the required time for convergence. Following table summarizes the average time required for different algorithms for finding a stationary point.

<p align="center">
  <img width="350" height="50" src="https://github.com/babakbarazandeh/Multi-step-Accelerated-Proximal-Gradient-Descent-Ascent/blob/master/table.png">
</p> <br/>

# Getting started
Run Main.m

