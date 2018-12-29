import numpy as np

cdef extern from "bsint_dust.c":
    int bsint_dust(double init[], double t, double dt, double beta,
                   double * final)


def bsint(double[:] init, double t, double dt, double beta):
    cdef double[6] final
    status = bsint_dust( & init[0], t, dt, beta, final)

    result = np.zeros(6)
    result[:] = final
    return result
