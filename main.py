# Importation

import numpy as np
import os
import time
from visualization import heatmap
import TSP_OpenCL_Paralellization as opencl_file
import TSP_Multiprocessing_Parallelization as multiprocessing_file
import TSP_No_Parallelization as nopar
import TSP_Threading_Parallelization as threading_file

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

# setting examples

np.random.seed(6)
NVilles = 10
TSP_method = 'GPU'  # different options are GPU, CPU, Threading, No_Par
rho_exp = 0.05
d_exp = 3
N_exp = 5000
alpha_exp = 0.99
init_exp = 1
dMat = np.matrix(np.random.randint(1, NVilles, size=(NVilles, NVilles)))  # Generation of a distance matrix
for i in range(NVilles):
    dMat[i, i] = 0

# Example of experiment
if __name__ == '__main__':
    t = time.time()
    print(dMat)
    if TSP_method == 'GPU':
        print("Execution of TSP_GPU")
        M = opencl_file.TSP_GPU(rho=rho_exp, d=d_exp, N=N_exp, distanceMatrix=dMat, alpha=alpha_exp, init=init_exp)
    elif TSP_method == "CPU":
        print("Execution of CPU_TSP")
        M = multiprocessing_file.TSP_process_parallel(rho=rho_exp, d=d_exp, N=N_exp,
                                                      distanceMatrix=dMat, alpha=alpha_exp, init=init_exp)
    elif TSP_method == "Threading":
        print("Execution of Thread based TSP")
        M = threading_file.TSP_thread_parallel_partial(rho=rho_exp, d=d_exp, N=N_exp,
                                                       distanceMatrix=dMat, alpha=alpha_exp, init=init_exp)
    elif TSP_method == "No_Par":
        print("Execution of TSP without parallelization")
        M = nopar.TSP(rho=rho_exp, d=d_exp, N=N_exp,
                      distanceMatrix=dMat, alpha=alpha_exp, init=init_exp)
    else:
        raise NameError('Please select a correct method of generation')
    print(time.time() - t)
    heatmap(M, [str(i) for i in range(10)])
