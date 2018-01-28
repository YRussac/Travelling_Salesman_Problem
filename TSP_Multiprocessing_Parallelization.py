# Importation
import math
import numpy as np
import time
import multiprocessing
import TSP_No_Parallelization as nopar


# Main function for multiprocessing-parallelized code


def paths_parallel(l, init, N, distanceMatrix, transition_matrix, d_paths, d_score):
    """
    :param l: Index of the process
    :param init: First point of the path
    :param N: Number of paths simulated
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param transition_matrix: The Markov transition matrix
    :param d_paths: Empty dictionary for communication with process
    :param d_score: Empty dictionary for communication with process
    """
    np.random.seed()
    transition_Matrix = transition_matrix.copy()
    pathsMatrix = nopar.random_multi_path(transition_Matrix, init, N)
    cost_paths = nopar.cost_multi_path(distanceMatrix, pathsMatrix)
    output_score = cost_paths
    d_paths[l] = pathsMatrix
    d_score[l] = output_score
    return None


def update_count_parallel(l, gamma, cost_paths, pathsMatrix, d_count, d_length):
    """
    :param l: Index of the process
    :param gamma: Threshold
    :param cost_paths: Matrix of the score of each simulated paths_kept
    :param pathsMatrix: Matrix containing the N previously simulated paths
    :param d_count: Empty dictionnary for communication with the process
    :param d_length: Empty dictionnary for communication with the process
    """
    n = pathsMatrix.shape[1]-1  # number of cities
    output_count = np.matrix(np.zeros((n,n)))
    paths_kept = np.where(cost_paths <= gamma)[1]
    for k in paths_kept:
        for i in range(pathsMatrix[k].shape[1]-1):
            output_count[pathsMatrix[k, i], pathsMatrix[k, i + 1]] += 1
    d_count[l] = output_count
    d_length[l] = len(paths_kept)
    return None


def TSP_process_parallel(rho, d, N, distanceMatrix, alpha, init, timer=False):
    """
    :param rho: A cross entropy parameter
    :param d: The number of times we want gamma to be the same, higher values should give more optimal paths but
    more computations
    :param N: Number of generated paths at each updating phase
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param alpha: A parameter for the cross-entropy
    :param init: The initial point for all the cities. We are searching an optimal solution starting from this point
    :param timer: If timer is true, stop after first iteration and return time
    :return: For the moment the function returns the transition matrix at the end of the updating phase
    """
    if timer:
        t0 = time.time()
    n = distanceMatrix.shape[0] # number of cities
    transition_Matrix = 1/(n-1)*np.matrix(np.ones((n, n))) - 1/(n-1)*np.identity(n)
    gamma_list = []
    t0 = time.time()
    i = 0
    while not (nopar.gamma_stable(gamma_list, d)):
        print('Iteration {}: {}'.format(i, time.time() - t0))
        print(gamma_list)
        i += 1
        manager = multiprocessing.Manager()
        d_count = manager.dict()
        d_length = manager.dict()
        d_score = manager.dict()
        d_paths = manager.dict()
        # generate paths
        jobs = [multiprocessing.Process(target=paths_parallel, args=(j, init, N // 4, distanceMatrix, transition_Matrix,
                                                                     d_paths, d_score)) for j in range(4)]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        # update gamma
        print('Begin update gamma {}'.format(time.time() - t0))
        ordered_scores = np.sort(np.concatenate([d_score[i] for i in d_score.keys()], axis=1))
        Gamma = ordered_scores[0, math.ceil(rho * N)]
        gamma_list.append(Gamma)

        print('End update gamma {}'.format(nopar.time.time() - t0))
        # generate counts for updating transition matrix
        jobs = [multiprocessing.Process(target=update_count_parallel, args=(j, Gamma, d_score[j], d_paths[j], d_count,
                                                                            d_length)) for j in range(4)]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
        numerator = sum([d_count[i] for i in d_count.keys()])
        denominator = sum([d_length[i] for i in d_length.keys()])
        transition_Matrix = (1-alpha)*transition_Matrix + alpha*(numerator/denominator)
        if timer:
            return (time.time() - t0)
    return transition_Matrix
