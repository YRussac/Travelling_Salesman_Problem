# Importation
import numpy as np
import os
import time
from visualization import heatmap, draw_path
import TSP_OpenCL_Paralellization as opencl_file
import TSP_Multiprocessing_Parallelization as multiprocessing_file
import TSP_No_Parallelization as nopar
import TSP_Threading_Parallelization as threading_file

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

# setting examples

np.random.seed(6)
NVilles = 120
loc = [(np.random.uniform(0,1), np.random.uniform(0,1)) for i in range(NVilles)]
TSP_method = 'GPU'  # different options are GPU, CPU, Threading, No_Par
rho_exp = 0.2
d_exp = 3
N_exp = 30000
alpha_exp = 0.95
init_exp = 1
dMat = np.matrix(np.zeros((NVilles, NVilles)))  # Generation of a distance matrix
for i in range(NVilles):
    for j in range(NVilles):
        dMat[i,j] = np.sqrt((loc[i][0] - loc[j][0])**2 + (loc[i][1] - loc[j][1])**2)

# Example of experiment
if __name__ == '__main__':
    t = time.time()
    print(dMat)
    if TSP_method == 'GPU':
        print("Execution of TSP_GPU")
        M = opencl_file.TSP_GPU(rho=rho_exp, d=d_exp, N=N_exp, distanceMatrix=dMat,
                                alpha=alpha_exp, init=init_exp,
                                graphics=False, loc=loc)
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
    heatmap(M, [str(i) for i in range(NVilles)])
    draw_path(loc, M, 1)
