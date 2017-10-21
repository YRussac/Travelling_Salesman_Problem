# importation
import numpy as np
import math
import multiprocessing
import queue as qu
import time
from visualization import heatmap

# setting examples
np.random.seed(6)
NVilles = 20
P = 1/(NVilles-1)*np.matrix(np.ones((NVilles, NVilles)))-1/(NVilles-1)*np.identity(NVilles)
dMat = np.matrix(np.random.randint(1,NVilles,size=(NVilles,NVilles)))
for i in range(NVilles):
    dMat[i,i] = 0

# Main function for non-parallelized


def cost_function(distanceMatrix, path):
    """
    :param distanceMatrix:  The matrix giving the distance between different points in the graph
    :param path: A path
    :return: The cost of the path (can be the total distance on for the tour)
    """
    res = 0
    for i in range(path.shape[1]-1):
        res += distanceMatrix[path[0,i], path[0,i+1]]
    res += distanceMatrix[path[0,-1], path[0,0]]
    return res


def cost_multi_path(distanceMatrix, pathsMatrix):
    """
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param pathsMatrix: A matrix where each row is a path
    :return: A list of costs, where the i-th element of the list is the cost for the i-th path
    """
    res = []
    for i in range(pathsMatrix.shape[0]):
        res.append(cost_function(distanceMatrix, pathsMatrix[i,:]))
    return np.matrix(res)


def normalize(M):
    """
    :param M: A matrix
    :return: The matrix normalized
    """
    return M/M.sum(axis=1)


def random_path(P, trajectory):
    """
    :param P: A transition matrix
    :param trajectory: We generate a tour with the transition matrix
    :return: A path, be careful to use this function trajectory must be initialized with the initial number
    in form of a list.
    """
    P_copy = P.copy()
    n = P_copy.shape[1]
    if len(trajectory) == n:
        trajectory.append(trajectory[0])
        return trajectory
    else:
        P_copy = normalize(P_copy)
        P_copy[:, trajectory[-1]] = 0
        nextPlace = np.random.choice(list(range(n)), size=1, p=P_copy[trajectory[-1], :].tolist()[0])[0]
        trajectory.append(nextPlace)
        return random_path(P_copy, trajectory)


def random_multi_path(P, init, N):
    """
    :param P: A transition matrix
    :param init: The initial point for the different paths
    :param N: The number of path we want to generate
    :return: Return a matrix where each row is a path
    """
    res = []
    for i in range(N):
        res.append(random_path(P, [init]))
    return np.matrix(res)


def gamma_stable(gList, d):
    """
    :param gList: A list containing the different values for gamma
    :param d: The stability points
    :return: True if and only if the d last terms of the list are equals
    """
    if (len(gList) < d):
        return False
    else:
        return len(set(gList[-d:])) == 1


def transition_presence(path, i, j):
    """
    :param path: A given path (a tour)
    :param i: The beginning of the transition
    :param j: The end point of the transition
    :return: True if and only if there is a transition between i and j in the tour
    """
    res = False
    indice_value_i = np.where(path == i)[1][0]
    if indice_value_i == path.shape[1] -1:
        if path[0, 0] == j:
            res = True
    elif path[0, indice_value_i + 1] == j:
        res = True
    return res


def update_transition_matrix(transition_matrix, pathsMatrix, gamma, distanceMatrix, alpha):
    """
    :param transition_matrix: A transition Matrix
    :param pathsMatrix: A matrix which contains a path in each row
    :param gamma: One of the cross-entropy parameter
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param alpha: Other parameter of the cross-entropy
    :return: The transition_matrix updated according to the cross-entropy theory
    """
    transition_Matrix = transition_matrix.copy()
    n = distanceMatrix.shape[0]  # number of cities
    paths_kept = np.where(cost_multi_path(distanceMatrix, pathsMatrix) <= gamma)[1]
    for i in range(0, n):
        for j in range(0, n):
            update_numerator = 0
            for k in paths_kept:
                if transition_presence(pathsMatrix[k, :], i, j):
                    update_numerator += 1
            transition_Matrix[i, j] = (1-alpha)*transition_Matrix[i, j] + alpha*update_numerator/len(paths_kept)
    return transition_Matrix


def TSP(rho, d, N, distanceMatrix, alpha, init):
    """
    :param rho: A cross entropy parameter
    :param d: The number of times we want gamma to be the same, higher values should give more optimal paths but
    more computations
    :param N: Number of generated paths at each updating phase
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param alpha: A parameter for the cross-entropy
    :param init: The initial point for all the cities. We are searching an optimal solution starting from this point
    :return: For the moment the function returns the transition matrix at the end of the updating phase
    """
    n = distanceMatrix.shape[0] # number of cities
    transition_Matrix = 1/(n-1)*np.matrix(np.ones((n, n))) - 1/(n-1)*np.identity(n)
    gamma_list = []
    while not(gamma_stable(gamma_list, d)):
        pathsMatrix = random_multi_path(transition_Matrix, init, N)
        ordered_scores = np.sort(cost_multi_path(distanceMatrix, pathsMatrix=pathsMatrix))
        Gamma = ordered_scores[0, math.ceil(rho*N)]
        gamma_list += [Gamma]
        transition_Matrix = update_transition_matrix(transition_matrix=transition_Matrix,
                                                     pathsMatrix=pathsMatrix,
                                                     gamma=Gamma, distanceMatrix=distanceMatrix,
                                                     alpha=alpha)
    return transition_Matrix


def TSP_thread_parallel(rho, d, N, distanceMatrix, alpha, init):
    """
    :param rho: A cross entropy parameter
    :param d: The number of times we want gamma to be the same, higher values should give more optimal paths but
    more computations
    :param N: Number of generated paths at each updating phase
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param alpha: A parameter for the cross-entropy
    :param init: The initial point for all the cities. We are searching an optimal solution starting from this point
    :return: For the moment the function returns the transition matrix at the end of the updating phase
    """
    n = distanceMatrix.shape[0] # number of cities
    transition_Matrix = 1/(n-1)*np.matrix(np.ones((n, n))) - 1/(n-1)*np.identity(n)
    gamma_list = []
    Gamma = cost_function(distanceMatrix, np.matrix(random_path(transition_Matrix, [init])))
    t0 = time.time()
    i = 0
    while not(gamma_stable(gamma_list, d)):
        print('Iteration {}: {}'.format(i, time.time()-t0))
        print(gamma_list)
        i += 1
        manager = multiprocessing.Manager()
        d_count = manager.dict()
        d_length = manager.dict()
        d_score = manager.dict()
        jobs = [multiprocessing.Process(target=thread_counter,
                            args=(j,init, N//10, Gamma, transition_Matrix,
                            distanceMatrix, d_count, d_length, d_score))
                for j in range(10)]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        print(d_length)
        #pathsMatrix = random_multi_path(transition_Matrix, init, N)
        #ordered_scores = np.sort(cost_multi_path(distanceMatrix, pathsMatrix=pathsMatrix))
        ordered_scores = np.sort(np.concatenate([d_score[i] for i in d_score.keys()], axis=1))
        Gamma = ordered_scores[0, math.ceil(rho*N)]
        gamma_list.append(Gamma)
        numerator = sum([d_count[i] for i in d_count.keys()])
        denominator = sum([d_length[i] for i in d_length.keys()])
        transition_Matrix = (1-alpha)*transition_Matrix + alpha*(numerator/denominator)
    return transition_Matrix


def thread_counter(l, init, N, gamma, transition_matrix, distanceMatrix, d_count, d_length, d_score):
    np.random.seed()
    transition_Matrix = transition_matrix.copy()
    pathsMatrix = random_multi_path(transition_Matrix, init, N)
    n = distanceMatrix.shape[0]  # number of cities
    output_count = np.matrix(np.zeros((n,n)))
    cost_paths = cost_multi_path(distanceMatrix, pathsMatrix)
    output_score = cost_paths
    paths_kept = np.where(cost_paths <= gamma)[1]
    for k in paths_kept:
        for i in range(pathsMatrix[k].shape[1]-1):
            output_count[pathsMatrix[k,i], pathsMatrix[k,i+1]] += 1
    d_count[l] = output_count
    d_score[l] = output_score
    d_length[l] = len(paths_kept)
    return




# Example of experiment
if __name__ == '__main__':
    t = time.time()
    print(dMat)
    M = TSP_thread_parallel(rho=0.1, d=5, N=10000, distanceMatrix=dMat, alpha=0.99, init=1)
    print(time.time()-t)
    heatmap(M, [str(i) for i in range(10)])
