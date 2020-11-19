# Expectation-Maximization Algorithm for Mixed Linear Regression Problem
Implementation code for the Expectation-Maximization (EM) algorithm for solving mixed linear regression problem under Gaussian and Laplacain additive noise regime.
# Abstract 
Mixed linear regression (MLR) model is among the most exemplary statistical tools for modeling non-linear distributions using a mixture of linear models. When the additive noise in MLR model is Gaussian, Expectation-Maximization (EM) algorithm is  a widely-used algorithm  for  maximum likelihood estimation of MLR parameters. However, when noise is non-Gaussian, the steps of EM algorithm may not have closed-form update rules, which makes EM algorithm impractical. In this work, we study the maximum likelihood estimation of the parameters of MLR model when the additive noise has non-Gaussian distribution.  In particular, we consider the case that noise has Laplacian distribution and we first
show that unlike the the Gaussian case, the resulting sub-problems of EM algorithm in this case does not have closed-form update rule, thus preventing us from using EM in this case. To overcome this issue, we propose a new algorithm based on combining the alternating direction method of multipliers (ADMM)  with EM algorithm idea. Our numerical experiments show that our method outperforms the EM algorithm in  statistical accuracy and computational time in non-Gaussian noise case.   

# Summary of the proposed algorithm
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

