import numba as nb
import numpy as np
from time import time, process_time_ns
import timeit


# https://numba.pydata.org/numba-doc/dev/user/5minguide.html

# x = np.arange(10000).reshape(100, 100)

# N_runs = 1000

# def go_fast2(a): # Function is compiled to machine code when called the first time
#     trace = 0.0
#     for i in range(N_runs):
#         for i in range(a.shape[0]):   # Numba likes loops
#             for j in range(a.shape[1]):   # Numba likes loops
#                 trace += np.tanh(a[i, j]) # Numba likes NumPy functions
#     return a + trace              # Numba likes NumPy broadcasting

# start2 = process_time_ns()
# go_fast2(x)

# end2 = process_time_ns()

# print((end2 - start2)/1e6/N_runs, 'ms')

# @nb.jit(nopython=True, fastmath=True, parallel=True) # Set "nopython" mode for best performance, equivalent to @njit
# def go_fast3(a): # Function is compiled to machine code when called the first time
#     trace = 0.0
#     for i in range(N_runs + 1):
#         for i in range(a.shape[0]):   # Numba likes loops
#             for j in range(a.shape[1]):   # Numba likes loops
#                 trace += np.tanh(a[i, j]) # Numba likes NumPy functions
#     return a + trace              # Numba likes NumPy broadcasting

# # go_fast3(x)
# go_fast3(x)

# print("Threading layer chosen: %s" % threading_layer())

# @nb.njit(parallel=True)
# def do_sum_parallel(A):
#     # each thread can accumulate its own partial sum, and then a cross
#     # thread reduction is performed to obtain the result to return
#     n = len(A)
#     acc = 0.
#     for i in prange(n):
#         acc += np.sqrt(A[i])
#     return acc

# @nb.njit(parallel=True, fastmath=True)
# def do_sum_parallel_fast(A):
#     n = len(A)
#     acc = 0.
#     for i in prange(n):
#         acc += np.sqrt(A[i])
#     return acc

# go_fast3(x)
# start2 = process_time_ns()
# for i in range(N_runs + 1):
#     go_fast2(x)

# end2 = process_time_ns()

# print((end2 - start2)/1e6/N_runs, 'ms')


s = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
len_s = s.size

# print('len_s + 2', len_s + 2)
# print(s[(2:11)%len_s])
print(s[np.s_[2::1]])
