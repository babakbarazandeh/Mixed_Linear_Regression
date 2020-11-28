from helper import *
from mpi4py import MPI
import pickle
import argparse


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def parse_args():
    parser = argparse.ArgumentParser(description="Python implementation of EM for MLR problems.")

    parser.add_argument('--Iteration_Number', nargs="?", type=int, default=500)
    parser.add_argument('--Sample_Number', nargs="?",type=int, default=200)
    parser.add_argument('--std_noise', nargs="?",type=float, default= 1.0)
    parser.add_argument('--std_coeff', nargs="?",type=float, default= 1.0)
    parser.add_argument('--Component_Range',nargs="?", type=int, default=5)
    parser.add_argument('--Dimension_Range', nargs="?",type=int, default=3)
    parser.add_argument('--Repetition_Number', nargs="?",type=int, default=5)
    parser.add_argument('--is_Laplacian', nargs="?",type=int, default=0)
    parser.add_argument('--step_size', type=float, nargs="?",default=1e-2)
    parser.add_argument('--iteration_sgd', type=int, nargs="?", default=5)

    return parser.parse_args()

def calculator( N_itr, sd_noise,sd_coeff, laplace_, N, C, D,R,N_itr_SGD):

    Error_EM_total = {} ## Save the final error for repetition result in dictionary
    Time_EM_total = {} ## Save the final time for repetition result in dictionary

    K = C[rank]
    Table_EM_G = np.zeros((len(D), R))  ## Save the result of repetetion for each K and d and put the in the Error_EM_total
    Time_EM_G = np.zeros((len(D), R))  ## The same as above for time, after parallelization we do this part
    for j in range(len(D)): ## Loop over d
        d = D[j]


        for ri in range(R): # loop over repetetion
            BETA_init = np.random.normal(0, sd_coeff, [d,K]) # create an intial point for both EM and ADMM
            BETA_OPT = np.random.normal(0, sd_coeff, [d,K]) # create the ground-truth randomly
            X = np.random.normal(0, sd_coeff, [N, d]) # create the features randomly
            ### Start to create y (observation) using y = <X,beta> + noise
            p = 1 / K * np.ones((1, K)) ## Each beta is equaly weighted
            r = np.random.multinomial(1, p[0], size=N)

            temp_1 = np.matmul(r, np.transpose(BETA_OPT))
            temp_2 = np.multiply(X, temp_1)
            temp_3 = np.sum(temp_2, axis=1).reshape([N,1])

            if laplace_:
                y = temp_3 + laprandi(0, sd_noise, N)
            else:
                y = temp_3 + np.random.normal(0, sd_coeff, [N,1])

            y= y.reshape([N,1])
            ### End to create y

            # run EM for that specific k and d (in repetitions)
            [Error_EM, Time_EM] = EM_func_general(X, y, BETA_OPT, BETA_init, N_itr, K, sd_noise, laplace_,step_size,N_itr_SGD)

            val = np.min(Error_EM)
            indx = np.min(np.argmin(Error_EM))
            Table_EM_G[j, ri] = val  ## Save its performance after the alghorithm runs
            Time_EM_G[j, ri] = Time_EM[0,indx]



    ## Save the resutls
    Error_EM_total[str(rank) + " :" + " Total"] = Table_EM_G
    Error_EM_total[str(rank) + " :" + " mean"] = np.mean(Table_EM_G , axis = 1)
    Error_EM_total[str(rank) + " :" + " std"] = np.std(Table_EM_G , axis = 1)


    Time_EM_total[str(rank) + " :" + " Total"] = Time_EM_G
    Time_EM_total[str(rank) + " :" + " mean"] = np.mean(Time_EM_G , axis = 1)
    Time_EM_total[str(rank) + " :" + " std"] = np.std(Time_EM_G , axis = 1)

    return Error_EM_total,Time_EM_total




if __name__ == '__main__':




    args = parse_args()
    N_itr = args.Iteration_Number
    step_size = args.step_size
    sd_noise = args.std_noise
    sd_coeff = args.std_coeff
    laplace_ = args.is_Laplacian
    N = args.Sample_Number
    R = args.Repetition_Number
    C = [*range(2, args.Component_Range, 1)]
    D = [*range(1, args.Dimension_Range, 1)]
    N_itr_SGD = args.iteration_sgd
    print(N_itr_SGD)
    result_EM_error, result_EM_time = calculator(N_itr, sd_noise, sd_coeff, laplace_, N, C, D, R,N_itr_SGD)

    data_error = comm.gather(result_EM_error, root=0)
    data_time = comm.gather(result_EM_time, root=0)

    if rank == 0:
        with open('result_error.txt', "wb") as fp:  # Pickling
            pickle.dump(data_error, fp)
        with open('result_time.txt', "wb") as fp:  # Pickling
            pickle.dump(data_time, fp)

