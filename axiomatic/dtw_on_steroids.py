import numpy as np
from cffi import FFI

from base import QuestionMarkSymbol
 
ffi = FFI()

ffi.cdef("""
    void compute_dtw_matrices(int* in_d_matrix, int* out_s_matrix, int* out_r_matrix, int m, int n, int subsequence);
    """)

C = ffi.verify(r"""
#include <stdio.h>
#define C_CONT_INDEX(n, i, j) ((i)*(n) + (j))
#define MAT_INDEX(matrix, n, i, j) ((matrix)[C_CONT_INDEX(n, i, j)])
#define D(i, j) MAT_INDEX(in_d_matrix, n, i, j)
#define S(i, j) MAT_INDEX(out_s_matrix, n, i, j)
#define R(i, j) MAT_INDEX(out_r_matrix, n, i, j)

void compute_dtw_matrices(int* in_d_matrix, int* out_s_matrix, int* out_r_matrix, int m, int n, int subsequence) {
    S(0, 0) = D(0, 0);
    R(0, 0) = 1;
    int a = 0;
    int b = 0;
    
    for(a = 1; a < m; ++a) {
        S(a, 0) = D(a, 0) + S(a - 1, 0);
        R(a, 0) = 1 + R(a - 1, 0);
    }
    
    for (b = 1; b < n; ++b) {
        S(0, b) = subsequence ? D(0, b) : D(0, b) + S(0, b - 1);
        R(0, b) = subsequence ? 1 : 1 + R(0, b - 1);
    }

    for(a = 1; a < m; ++a) {
        for(b = 1; b < n; ++b) {
            int Dab = D(a, b);
            int Sa1b1 = S(a - 1, b - 1);
            int Ra1b1 = R(a - 1, b - 1);
            int Sab1 = S(a, b - 1);
            int Rab1 = R(a, b - 1);
            int Sa1b = S(a - 1, b);
            int Ra1b = R(a - 1, b);
            int diag = (Dab + Sa1b1) * (Ra1b1 + 1);
            int right = (Dab + Sab1) * (Rab1 + 1);
            int down = (Dab + Sa1b) * (Ra1b + 1);
            if (down < diag && down < right) {
                S(a, b) = Dab + Sa1b;
                R(a, b) = Ra1b + 1;
            } else if (diag <= down && diag <= right) {
                S(a, b) = Dab + Sa1b1;
                R(a, b) = Ra1b1 + 1;
            } else if (right < diag && right <= down) {
                S(a, b) = Dab + Sab1;
                R(a, b) = Rab1 + 1;
            }
        }
    }
}
""", verbose=True)

def dtw_distances(model, observed_marking):
    int_type = np.int32 if ffi.sizeof('int') == 4  else np.int64
    model = np.array(model)
    observed_marking = np.array(observed_marking)
    dist_matrix = np.logical_and(model.reshape(-1, 1) != observed_marking.reshape(1, -1), model.reshape(-1, 1) != QuestionMarkSymbol).astype(int_type, order='C')
    s_matrix = np.full(dist_matrix.shape, -1, dtype=int_type, order='C')
    r_matrix = np.full(dist_matrix.shape, -1, dtype=int_type, order='C')
    C.compute_dtw_matrices(ffi.cast("int*", dist_matrix.ctypes.data), ffi.cast("int*", s_matrix.ctypes.data), ffi.cast("int*", r_matrix.ctypes.data), dist_matrix.shape[0], dist_matrix.shape[1], 1)
    dist = s_matrix[-1, :].astype(float) / r_matrix[-1, :].astype(float)
    
    return dist