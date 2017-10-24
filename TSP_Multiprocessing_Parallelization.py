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


def paths_parallel(l, init, N, distanceMatrix, transition_matrix, d_paths, d_score):
    """
    :param l: Index of the process
    :param init: First point of the path
    :param N: Number of paths simulated
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param transition_matrix: The Markov transition matrix
    :param d_paths: Empty dictionnary for communication with process
    :param d_score: Empty dictionnary for communication with process
    """
    np.random.seed()
    transition_Matrix = transition_matrix.copy()
    pathsMatrix = random_multi_path(transition_Matrix, init, N)
    cost_paths = cost_multi_path(distanceMatrix, pathsMatrix)
    output_score = cost_paths
    d_paths[l] = pathsMatrix
    d_score[l] = output_score
    return


def update_count_parallel(l, gamma, cost_paths, pathsMatrix, d_count, d_length):
    """
    :param l: Index of the process
    :param gamma: Threshold
    :param cost_paths: Vector of the score of each simulated paths_kept
    :param pathsMatrix: Matrix containing the N previously simulated paths
    :param d_count: Empty dictionnary for communication with the process
    :param d_length: Empty dictionnary for communication with the process
    """
    n = pathsMatrix.shape[1]-1  # number of cities
    output_count = np.matrix(np.zeros((n,n)))
    paths_kept = np.where(cost_paths <= gamma)[1]
    for k in paths_kept:
        for i in range(pathsMatrix[k].shape[1]-1):
            output_count[pathsMatrix[k,i], pathsMatrix[k,i+1]] += 1
    d_count[l] = output_count
    d_length[l] = len(paths_kept)


def TSP_process_parallel(rho, d, N, distanceMatrix, alpha, init):
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
        d_paths = manager.dict()
        # generate paths
        jobs = [multiprocessing.Process(target=paths_parallel,
                            args=(j,init, N//4, distanceMatrix,
                            transition_Matrix,
                            d_paths, d_score))
                for j in range(4)]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        # update gamma
        ordered_scores = np.sort(np.concatenate([d_score[i] for i in d_score.keys()], axis=1))
        Gamma = ordered_scores[0, math.ceil(rho*N)]
        gamma_list.append(Gamma)

        # generate counts for updating transition matrix
        jobs = [multiprocessing.Process(target=update_count_parallel,
                            args=(j, Gamma, d_score[j], d_paths[j],
                            d_count, d_length))
                for j in range(4)]
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()

        numerator = sum([d_count[i] for i in d_count.keys()])
        denominator = sum([d_length[i] for i in d_length.keys()])
        transition_Matrix = (1-alpha)*transition_Matrix + alpha*(numerator/denominator)
    return transition_Matrix




# Example of experiment
if __name__ == '__main__':
    t = time.time()
    print(dMat)
    M = TSP_process_parallel(rho=0.05, d=3, N=50000, distanceMatrix=dMat, alpha=0.99, init=1)
    print(time.time()-t)
    heatmap(M, [str(i) for i in range(10)])
