# importation
import numpy as np
import math
import threading
import queue as qu
import TSP_No_Parallelization as nopar
import time


# setting examples
P = 1/9*np.matrix(np.ones((10, 10)))-1/9*np.identity(10)

# generation of a random transition matrix
dMat = np.matrix(np.random.randint(1,10,size=(10,10)))
for i in range(10):
    dMat[i, i] = 0

# Main function for non-parallelized

def thread_computation(init, N, gamma, transition_matrix, distanceMatrix, q_count, q_score):
    """
    This is the function which will be launched on every core of the machine
    We will generate N/4 path on each of the 4 processors and update the transition matrix
    :param init:
    :param N:
    :param gamma:
    :param transition_matrix:
    :param distanceMatrix:
    :param q_count:
    :param q_score:
    :return:
    """
    output_count = {}
    local_transition_matrix = transition_matrix.copy()
    pathsMatrix = nopar.random_multi_path(local_transition_matrix, init, N)
    n = distanceMatrix.shape[0]  # number of cities
    cost_paths = nopar.cost_multi_path(distanceMatrix, pathsMatrix)
    paths_kept = np.where(cost_paths <= gamma)[1]
    for i in range(n):
        for j in range(n):
            update_numerator = 0
            for k in paths_kept:
                if nopar.transition_presence(pathsMatrix[k, :], i, j):
                    update_numerator += 1
            output_count[(i, j)] = update_numerator, len(paths_kept)
    q_count.put(output_count)
    q_score.put(cost_paths)
    return


def TSP_thread_parallel_tot(rho, d, N, distanceMatrix, alpha, init):
    """
    In this version parallelization on both the generation of paths and on updating process
    :param rho: A cross entropy parameter
    :param d: The number of times we want gamma to be the same, higher values should give more optimal paths but
    more computations
    :param N: Number of generated paths at each updating phase
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param alpha: A parameter for the cross-entropy
    :param init: The initial point for all the cities. We are searching an optimal solution starting from this point
    :return: For the moment the function returns the transition matrix at the end of the updating phase
    """
    n = distanceMatrix.shape[0]  # number of cities
    transition_Matrix = 1/(n-1)*np.matrix(np.ones((n, n))) - 1/(n-1)*np.identity(n)
    gamma_list = []
    gamma = nopar.cost_function(distanceMatrix, np.matrix(nopar.random_path(transition_Matrix, [init])))
    while not(nopar.gamma_stable(gamma_list, d)):
        q_count = qu.Queue()
        q_score = qu.Queue()
        t1 = threading.Thread(name='first_thread', target=thread_computation,
                              args=(init, N//4, gamma,
                            transition_Matrix, distanceMatrix, q_count, q_score))
        t2 = threading.Thread(name='first_thread', target=thread_computation,
                            args=(init, N//4, gamma,
                            transition_Matrix,distanceMatrix, q_count, q_score))
        t3 = threading.Thread(name='first_thread', target=thread_computation,
                            args=(init, N//4, gamma,
                            transition_Matrix, distanceMatrix, q_count, q_score))
        t4 = threading.Thread(name='first_thread', target=thread_computation,
                            args=(init, N//4, gamma,
                            transition_Matrix, distanceMatrix, q_count, q_score))
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        output_score1 = q_score.get()
        output_score2 = q_score.get()
        output_score3 = q_score.get()
        output_score4 = q_score.get()
        output_count1 = q_count.get()
        output_count2 = q_count.get()
        output_count3 = q_count.get()
        output_count4 = q_count.get()
        ordered_scores = np.matrix(np.sort(np.concatenate((output_score1, output_score2,
                                                           output_score3, output_score4), axis=1)))
        Gamma = ordered_scores[0, math.ceil(rho*N)]
        gamma_list.append(Gamma)
        for i,j in list(output_count1.keys()):
            numerator = output_count1[(i, j)][0] + output_count2[(i, j)][0] + output_count3[(i, j)][0] \
                        + output_count4[(i, j)][0]
            denominator = output_count1[(i,j)][1] + output_count2[(i,j)][1] + output_count3[(i, j)][1]\
                          + output_count4[(i, j)][1]
            transition_Matrix[i, j] = (1-alpha)*transition_Matrix[i,j] + alpha*numerator/denominator
    return transition_Matrix


def partial_thread_computation(pathMatrix, cost_paths,gamma, q_count):
    """
    This is the function which will be launched on every core of the machine.
    We will update the transition matrix with N//4 paths on every cores
    :param pathMatrix: A matrix containing some paths
    :param cost_paths: The costs of the associated paths
    :param gamma: Parameter used for the cross entropy
    :param q_count: A queue used during the threading procedure
    :return: Nothing but the q_count queue have a matrix which is appended where we have the required value
    for the updating process
    """
    output_count = {}
    n = pathMatrix.shape[1]-1  # difference between number of cities and a tour
    paths_kept = np.where(cost_paths <= gamma)[1]
    for i in range(n):
        for j in range(n):
            update_numerator = 0
            for k in paths_kept:
                if nopar.transition_presence(pathMatrix[k, :], i, j):
                    update_numerator += 1
            output_count[(i, j)] = update_numerator, len(paths_kept)  # check indentation here
    q_count.put(output_count)
    return None


def TSP_thread_parallel_partial(rho, d, N, distanceMatrix, alpha, init):
    """
    In this version parallelization on updating process
    :param rho: A cross entropy parameter
    :param d: The number of times we want gamma to be the same, higher values should give more optimal paths but
    more computations
    :param N: Number of generated paths at each updating phase
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param alpha: A parameter for the cross-entropy
    :param init: The initial point for all the paths. We are searching an optimal solution starting from this point
    :return: For the moment the function returns the transition matrix at the end of the updating phase
    """
    n = distanceMatrix.shape[0]  # number of cities
    print("Number of cities in the TSP_partial algorithm = " + str(n))
    transition_Matrix = 1/(n-1)*np.matrix(np.ones((n, n))) - 1/(n-1)*np.identity(n)
    gamma_list = []
    while not(nopar.gamma_stable(gamma_list, d)):
        q_count = qu.Queue()
        pathsMatrix = nopar.random_multi_path(transition_Matrix, init, N)
        mat1 = pathsMatrix[0:N//4, :]
        mat2 = pathsMatrix[N//4: 2*N//4, :]
        mat3 = pathsMatrix[2*N//4: 3*N//4, :]
        mat4 = pathsMatrix[3*N//4:, :]
        cost_paths = nopar.cost_multi_path(distanceMatrix, pathsMatrix=pathsMatrix)
        cout1 = cost_paths[0, 0:N//4]
        cout2 = cost_paths[0, N//4: 2*N//4]
        cout3 = cost_paths[0, 2*N//4: 3*N//4]
        cout4 = cost_paths[0, 3*N//4:]
        ordered_scores = np.sort(cost_paths)
        Gamma = ordered_scores[0, math.ceil(rho*N)]
        gamma_list.append(Gamma)
        print(gamma_list)
        t1 = threading.Thread(name='first_thread', target=partial_thread_computation,
                              args=(mat1, cout1, Gamma, q_count))
        t2 = threading.Thread(name='second_thread', target=partial_thread_computation,
                              args=(mat2, cout2, Gamma, q_count))
        t3 = threading.Thread(name='third_thread', target=partial_thread_computation,
                              args=(mat3, cout3, Gamma, q_count))
        t4 = threading.Thread(name='forth_thread', target=partial_thread_computation,
                              args=(mat4, cout4, Gamma, q_count))
        t1.start()
        t2.start()
        t3.start()
        t4.start()
        t1.join()
        t2.join()
        t3.join()
        t4.join()
        output_count1 = q_count.get()
        output_count2 = q_count.get()
        output_count3 = q_count.get()
        output_count4 = q_count.get()
        for i,j in list(output_count1.keys()):
            numerator = output_count1[(i, j)][0] + output_count2[(i, j)][0] + output_count3[(i, j)][0] \
                        + output_count4[(i, j)][0]
            denominator = output_count1[(i, j)][1] + output_count2[(i,j)][1] + output_count3[(i, j)][1]\
                          + output_count4[(i, j)][1]
            transition_Matrix[i, j] = (1-alpha)*transition_Matrix[i,j] + alpha*numerator/denominator
    return transition_Matrix




# Example of experiment
if __name__ == '__main__':
    t2 = time.time()
    print(TSP_thread_parallel_partial(rho=0.1, d=5, N=8000, distanceMatrix=dMat, alpha=0.99, init=1))
    print(time.time() - t2)



# t = time.time()
# print(nopar.TSP(rho=0.1, d=5, N=8000, distanceMatrix=dMat, alpha=0.99, init=1))
# print(time.time()-t)

# t3 = time.time()
# print(TSP_thread_parallel_tot(rho=0.1, d=5, N=8000, distanceMatrix=dMat, alpha=0.99, init=1))
# print(time.time()-t3)



