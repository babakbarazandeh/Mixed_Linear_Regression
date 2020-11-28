import numpy as np
from math import *
import numpy.matlib
from scipy.optimize import linear_sum_assignment
import time



def Lpdf(x,mu,sigma):
    b = sigma / sqrt(2)

    y = (1/(2 * b)) * np.exp((-1 / b) * np.abs((x - mu)))
    return y

def laprandi(mu, sigma, N):
    z = np.random.uniform(0, 1, [N,1])
    b = sigma / sqrt(2)
    y = mu - b * np.multiply(np.sign(z - 0.5),np.log(1 - 2 * np.abs(z - 0.5)))
    return y

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*pi*var)**.5
    temp = np.power(x-mean,2)
    num = np.exp(-temp)/(2*var)
    return num/denom
def distance(BETA,BETA_OPT):
    K = BETA.shape[1]
    dist = np.zeros((K, K))

    for i_ in range(K):
        for j_ in range(K):
            dist[i_, j_] = np.linalg.norm(BETA[:, i_] - BETA_OPT[:, j_]) / np.linalg.norm(BETA_OPT[:, j_])

    row_ind, col_ind = linear_sum_assignment(dist)
    cost = dist[row_ind, col_ind].sum()

    return cost

def weight(X, BETA,y,sd_noise, laplace_):
    Z = np.matmul(X, BETA)
    temp = np.matlib.repmat(y, 1, BETA.shape[1])

    if laplace_:
        g1 = Lpdf(temp, Z, sd_noise)
    else:
        g1 = normpdf(temp, Z, sd_noise)

    g2 = np.sum(g1, 1)
    g2 = g2.reshape([len(g2), 1])
    g2 = np.matlib.repmat(g2,1,g1.shape[1])
    w = np.divide(g1, g2)

    return w



def solver(x_0, X,y,w, step_size,N_itr_SGD):
    t = np.zeros((y.shape[0], y.shape[0]))
    for i in range(N_itr_SGD):
        temp1 = y.T - np.matmul(X,x_0)
        temp1 = np.sign(temp1)
        np.fill_diagonal(t, temp1)
        grad = np.matmul(X.T,t)
        grad = - np.matmul(grad, w)
        x_0 = x_0 - step_size * grad/(np.linalg.norm(grad))

    return x_0



def EM_func_general( X, y, BETA_OPT,BETA,  N_itr,K,sd_noise, laplace_,step_size,N_itr_SGD):
    Error = np.ones((1, N_itr))
    Time = np.zeros((1, N_itr))


    d = X.shape[1]
    Error[0,0] = distance(BETA, BETA_OPT)

    tic = time.perf_counter()
    for it in range(1,N_itr,1):

        w = weight(X, BETA,y,sd_noise, laplace_)

        for k in range(K):
            TEMP = np.matlib.repmat(w[:,[k]],1,d)

            if laplace_:

                BETA[:, [k]] = solver(BETA[:, [k]], X, y,w[:,[k]],step_size,N_itr_SGD)
            else:
                temp1 = np.multiply(X,TEMP)
                temp1 = np.matmul(X.T,temp1)
                temp1 = np.linalg.inv(temp1)
                temp2 = np.multiply(w[:,[k]],y)
                temp2 = np.matmul(X.T,temp2)
                BETA[:,[k]] = np.matmul(temp1,temp2)

        Error[0, it] = distance(BETA, BETA_OPT)
        Time[0,it] = time.perf_counter() -tic
    return [Error, Time]





