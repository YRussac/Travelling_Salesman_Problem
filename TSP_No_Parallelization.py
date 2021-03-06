# Importation
import math
import numpy as np
import time


# Main function for non-parallelized code


def cost_function(distanceMatrix, path):
    """
    :param distanceMatrix:  The matrix giving the distance between different points in the graph
    :param path: A path
    :return: The cost of the path (can be the total distance on for the tour)
    """
    res = 0
    for i in range(path.shape[1]-1):
        try:
            res += distanceMatrix[path[0, i], path[0, i + 1]]
        except:
            print(i)
            print(i+1)
            print(path[0,:])
    res += distanceMatrix[path[0, -1], path[0, 0]]
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
    if len(gList) < d:
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
    output_count = np.matrix(np.zeros((n,n)))
    paths_kept = np.where(cost_multi_path(distanceMatrix, pathsMatrix) <= gamma)[1]
    for k in paths_kept:
        for i in range(pathsMatrix[k].shape[1]-1):
            output_count[pathsMatrix[k, i], pathsMatrix[k, i + 1]] += 1
    transition_Matrix = (1-alpha)*transition_Matrix + alpha*output_count/len(paths_kept)
    return transition_Matrix


def TSP(rho, d, N, distanceMatrix, alpha, init, timer=False):
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
    i = 0
    t0 = time.time()
    while not(gamma_stable(gamma_list, d)):
        print('Iteration {}: {}'.format(i, time.time()-t0))
        print(gamma_list)
        i += 1
        pathsMatrix = random_multi_path(transition_Matrix, init, N)
        ordered_scores = np.sort(cost_multi_path(distanceMatrix, pathsMatrix=pathsMatrix))
        print(ordered_scores.shape)
        Gamma = ordered_scores[0, math.ceil(rho*N)]
        if len(gamma_list) == 0:
            gamma_list.append(Gamma)
        elif (Gamma <= gamma_list[-1]):
            gamma_list.append(Gamma)
        transition_Matrix = update_transition_matrix(transition_matrix=transition_Matrix,
                                                     pathsMatrix=pathsMatrix,
                                                     gamma=Gamma, distanceMatrix=distanceMatrix,
                                                     alpha=alpha)
        if timer:
            return (time.time() - t0)
    return transition_Matrix
