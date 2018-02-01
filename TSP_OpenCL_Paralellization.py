# Importation
import math
import pyopencl as cl
import numpy as np
import time
import TSP_No_Parallelization as nopar
from visualization import heatmap, draw_path


##  Main function for GPU-parallelized code


def random_multi_path_GPU(P, init, N):
    """
    :param P: The transition matrix
    :param init: The initial city for the different paths (and the last city visited because we are creating tours)
    :param N: The number of generated paths
    :return: N paths generated using the transition matrix P
    """
    n = P.shape[0]
    U = np.random.uniform(0, 1, size=(n + 1) * N)
    U = U.astype(np.float32)
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
    res_np = np.zeros((N, n + 1), dtype=np.int32)

    # Setting the context for OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    mf = cl.mem_flags

    # Creating the different buffers
    # Those buffers will "feed" the GPUs
    U_buf = cl.Buffer(context, mf.COPY_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=U)
    P_buf = cl.Buffer(context, mf.COPY_HOST_PTR | mf.COPY_HOST_PTR, hostbuf=P)
    res_buf = cl.Buffer(context, mf.WRITE_ONLY, res_np.nbytes)
    Pbis_buf = cl.Buffer(context, mf.WRITE_ONLY, Pbis.nbytes)
    current_buf = cl.Buffer(context, mf.WRITE_ONLY, current.nbytes)
    cNorm_buf = cl.Buffer(context, mf.WRITE_ONLY, cNorm.nbytes)
    cumul_buf = cl.Buffer(context, mf.WRITE_ONLY, cumul.nbytes)
    z_buf = cl.Buffer(context, mf.WRITE_ONLY, z.nbytes)

    # Here is the code that will run the GPU computations
    ### BEGIN GPU ###
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
        while(cumul[i] < U[i*n+j]){
            z[i]++;
            cumul[i] += Pbis[i*n*n + current[i]*n + z[i]];
        }
        current[i] = z[i];
      }
      R[i*(n+1) + n] = init;
        }
    """).build()

    # run the previous OpenCL function with our buffers as input
    program.generate_paths(queue, (res_np.shape[0],), None, U_buf, P_buf, np.uint16(n), np.uint16(N), np.uint16(init),
                           res_buf, Pbis_buf, current_buf, cNorm_buf, cumul_buf, z_buf)
    chem_gen = np.empty_like(res_np)
    # get the result with the enqueue functon
    cl.enqueue_copy(queue, chem_gen, res_buf)
    ### END GPU ###
    
    return np.matrix(chem_gen)


def TSP_GPU(rho, d, N, distanceMatrix, alpha, init, timer=False, graphics=False, loc=[]):
    """
    :param rho: A cross entropy parameter
    :param d: The number of times we want gamma to be the same, higher values should give more optimal paths but
    more computations
    :param N: Number of generated paths at each updating phase
    :param distanceMatrix: The matrix giving the distance between different points in the graph
    :param alpha: A parameter for the cross-entropy
    :param init: The initial point for all the cities. We are searching an optimal solution starting from this point
    :param timer: If timer is true, stop after first iteration and return time
    :param graphics: If true, show graphics after each iteration
    :param loc: localisation of the cities
    :return: For the moment the function returns the transition matrix at the end of the updating phase
    """
    if timer:
        t0 = time.time()
    n = distanceMatrix.shape[0] # number of cities
    transition_Matrix = 1/(n-1)*np.matrix(np.ones((n, n))) - 1/(n-1)*np.identity(n)
    gamma_list = []
    i = 0
    t = time.time()

    if graphics:
        heatmap(transition_Matrix, [str(i) for i in range(transition_Matrix.shape[0])], 'Transition matrix heatmap')
        draw_path(loc, transition_Matrix, 1)

    while not (nopar.gamma_stable(gamma_list, d)):
        print('Iteration {}: {}'.format(i, time.time() - t))
        print('Evolution de la liste de Gamma:' + str(gamma_list))
        i += 1

        ### BEGIN GPU ###
        # Se function random_multi_path to see in
        # details the border between GPU and CPU
        pathsMatrix = random_multi_path_GPU(transition_Matrix, init, N)
        ### END GPU ###
        # print('random_multi_paths {}: {}'.format(i, nopar.time.time()-t0))
        ordered_scores = np.sort(nopar.cost_multi_path(distanceMatrix, pathsMatrix=pathsMatrix))
        Gamma = ordered_scores[0, math.ceil(rho * N)]
        if len(gamma_list) == 0:
            gamma_list.append(Gamma)
        elif (Gamma <= gamma_list[-1]):
            gamma_list.append(Gamma)
        transition_Matrix = nopar.update_transition_matrix(transition_matrix=transition_Matrix,
                                                           pathsMatrix=pathsMatrix, gamma=Gamma,
                                                           distanceMatrix=distanceMatrix, alpha=alpha)
        if timer:
            return (time.time() - t0)
        if graphics:
            heatmap(transition_Matrix, [str(i) for i in range(transition_Matrix.shape[0])], 'Transition matrix heatmap')
            draw_path(loc, transition_Matrix, 1)
    return transition_Matrix
