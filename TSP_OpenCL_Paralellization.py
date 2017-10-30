import numpy as np
import pyopencl as cl
import numpy.linalg as la

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

def random_multi_path_gpu(P, init, N):
    n = P.shape[0]
    U = np.random.uniform(0,1, size=(n+1)*N)
    U = U.astype(np.float32)
    P = np.matrix(1/(n-1)*np.ones((n,n)) - 1/(n-1)*np.identity(n))
    P = P.astype(np.float32)
    Pbis = np.matrix(np.zeros((N*n, n)))
    Pbis = Pbis.astype(np.float32)
    current = np.array([0 for i in range(N)])
    current = current.astype(np.int32)
    cNorm = np.array([0. for i in range(N)])
    cNorm = cNorm.astype(np.float32)
    cumul = np.array([0. for i in range(N)])
    cumul = cumul.astype(np.float32)
    z = np.array([0 for i in range(N)])
    z = z.astype(np.int32)
    res_np = np.zeros((N, n+1),dtype = np.int32)

    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    mf = cl.mem_flags

    U_buf = cl.Buffer(context, mf.COPY_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=U)
    P_buf = cl.Buffer(context, mf.COPY_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=P)
    res_buf = cl.Buffer(context, mf.WRITE_ONLY, res_np.nbytes)
    Pbis_buf = cl.Buffer(context, mf.WRITE_ONLY, Pbis.nbytes)
    current_buf = cl.Buffer(context, mf.WRITE_ONLY, current.nbytes)
    cNorm_buf = cl.Buffer(context, mf.WRITE_ONLY, cNorm.nbytes)
    cumul_buf = cl.Buffer(context, mf.WRITE_ONLY, cumul.nbytes)
    z_buf = cl.Buffer(context, mf.WRITE_ONLY, z.nbytes)

    init = 0

    program = cl.Program(context, """
    __kernel void generate_paths(__global const float *U, __global float *P, ushort n,
        ushort N, ushort init, __global int *R, __global float *Pbis,
        __global int *current, __global float *cNorm, __global float *cumul, __global int *z){
      int i = get_global_id(0);
      for (int k = 0; k < n; k++){
          for (int l = 0; l < n; l++){
              Pbis[i*n*n + k*n + l] = P[k*n + l];
          }
      }
      current[i] = init;
      for(int j = 0; j < n; j++){
        R[i*(n+1) + j] = current[i];
        for (int l = 0; l<n; l++){
          Pbis[i*n*n + l*n + current[i]] = 0;
        }
        cNorm[i] = 0.;
        for (int l = 0; l<n; l++){
          cNorm[i] += Pbis[i*n*n + current[i]*n + l];
        }
        for (int l = 0; l<n; l++){
          Pbis[i*n*n + current[i]*n + l] = Pbis[i*n*n + current[i]*n + l]/cNorm[i];
        }

        cumul[i] = 0.;
        z[i] = 0;
        cumul[i] += Pbis[i*n*n + current[i]*n + z[i]];
        while(cumul[i] <= U[i*n+j]){
            z[i]++;
            cumul[i] += Pbis[i*n*n + current[i]*n + z[i]];
        }
        current[i] = z[i];
      }
      R[i*(n+1) + n] = init;
        }
    """).build()

    program.generate_paths(queue, (res_np.shape[0],), None, U_buf, P_buf, np.uint16(n),np.uint16(N),np.uint16(init), res_buf, Pbis_buf, current_buf, cNorm_buf, cumul_buf, z_buf)
    chem_gen = np.empty_like(res_np)
    cl.enqueue_copy(queue, chem_gen, res_buf)
    return np.matrix(chem_gen)

# importation
import numpy as np
import math
from visualization import heatmap
import time

# setting examples
np.random.seed(6)
NVilles = 20
P = 1/(NVilles-1)*np.matrix(np.ones((NVilles, NVilles)))-1/(NVilles-1)*np.identity(NVilles)
# Generation of a distance matrix reproductible

dMat = np.matrix(np.random.randint(1, NVilles, size=(NVilles, NVilles)))
for i in range(NVilles):
    dMat[i, i] = 0

# Main function for non-parallelized


def cost_function(distanceMatrix, path):
    """
    :param distanceMatrix:  The matrix giving the distance between different points in the graph
    :param path: A path
    :return: The cost of the path (can be the total distance on for the tour)
    """
    res = 0
    for i in range(path.shape[1]-1):
        res += distanceMatrix[path[0, i], path[0, i + 1]]
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
            output_count[pathsMatrix[k,i], pathsMatrix[k,i+1]] += 1
    transition_Matrix = (1-alpha)*transition_Matrix + alpha*output_count/len(paths_kept)
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
    i = 0
    t0 = time.time()
    while not(gamma_stable(gamma_list, d)):
        print('Iteration {}: {}'.format(i, time.time()-t0))
        print(gamma_list)
        i += 1
        pathsMatrix = random_multi_path_gpu(transition_Matrix, init, N)
        print('random_multi_paths {}: {}'.format(i, time.time()-t0))
        ordered_scores = np.sort(cost_multi_path(distanceMatrix, pathsMatrix=pathsMatrix))
        print(ordered_scores.shape)
        Gamma = ordered_scores[0, math.ceil(rho*N)]
        gamma_list.append(Gamma)
        transition_Matrix = update_transition_matrix(transition_matrix=transition_Matrix,
                                                     pathsMatrix=pathsMatrix,
                                                     gamma=Gamma, distanceMatrix=distanceMatrix,
                                                     alpha=alpha)
    return transition_Matrix


# Example of experiment
if __name__ == '__main__':
    n = 4
    transition_Matrix = np.matrix('0. 0. 1. 0.; 0. 0. 0. 1.; 0. 1. 0. 0.; 1. 0. 0. 0.')
    print(random_multi_path_gpu(transition_Matrix, 0, 5))
    print(random_multi_path(transition_Matrix, 0, 5))
    '''print("Execution of TSP_No_Parallelization")
    t = time.time()
    print(dMat)
    M = TSP(rho=0.05, d=3, N=50000, distanceMatrix=dMat, alpha=0.99, init=1)
    print(time.time() - t)
    heatmap(M, [str(i) for i in range(10)])'''
