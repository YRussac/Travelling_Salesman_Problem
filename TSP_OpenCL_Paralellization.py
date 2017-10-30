import numpy as np
import pyopencl as cl
import numpy.linalg as la

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'

(n, N) = (50,50000)
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
print("Platform Selected = %s" %platform.name)
print("Device Selected = %s" %device.name)
print("Les chemins générés sont:")
print(chem_gen)
