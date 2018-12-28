# distutils: sources =
from libc.math cimport sqrt
cimport cbsint

cdef int grav2_rp(double t, const double y[], double f[], void * params):
    """Two-body problem with radiation pressure."""
    (void)(t)  # avoid unused parameter warning

    cdef double r2 = y[0] * y[0] + y[1] * y[1] + y[2] + y[2]
    cdef double r1 = sqrt(r2)
    cdef double r3 = r1 * r2
    cdef double beta = * < double * >params
    cdef double mu_r3 = MU / r3 * (1 - beta)

    f[0] = y[3]
    f[1] = y[4]
    f[2] = y[5]
    f[3] = -y[0] * mu_r3
    f[4] = -y[1] * mu_r3
    f[5] = -y[2] * mu_r3

    return cbsint.GSL_SUCCESS

cdef int jac_grav2_rp(double t, const double y[], double * dfdy,
                      double dfdt[], void * params):
    """Jacobian for grav2_rp."""
    (void)(t)  # avoid unused parameter warning

    cdef double r2 = y[0] * y[0] + y[1] * y[1] + y[2] + y[2]
    cdef double r1 = sqrt(r2)
    cdef double r3 = r1 * r2
    cdef double beta = * < double * >params
    cdef double mu_r5 = MU * (1 - beta) / r3 / r2

    cdef cbsint.gsl_matrix_view dfdy_mat = cbsint.gsl_matrix_view_array(
        dfdy, 6, 6)

    cdef cbsint.gsl_matrix * m = &dfdy_mat.matrix
    cbsint.gsl_matrix_set_zero(m)

    cbsint.gsl_matrix_set(m, 0, 3, 1.0)
    cbsint.gsl_matrix_set(m, 1, 4, 1.0)
    cbsint.gsl_matrix_set(m, 2, 5, 1.0)

    cbsint.gsl_matrix_set(m, 3, 0, mu_r5 * (r2 - 3 * y[0] * y[0]))
    cbsint.gsl_matrix_set(m, 3, 1, -mu_r5 * 3 * y[0] * y[1])
    cbsint.gsl_matrix_set(m, 3, 2, -mu_r5 * 3 * y[0] * y[2])

    cbsint.gsl_matrix_set(m, 4, 0, -mu_r5 * 3 * y[1] * y[0])
    cbsint.gsl_matrix_set(m, 4, 1, mu_r5 * (r2 - 3 * y[1] * y[1]))
    cbsint.gsl_matrix_set(m, 4, 2, -mu_r5 * 3 * y[1] * y[2])

    cbsint.gsl_matrix_set(m, 5, 0, -mu_r5 * 3 * y[2] * y[0])
    cbsint.gsl_matrix_set(m, 5, 1, -mu_r5 * 3 * y[2] * y[1])
    cbsint.gsl_matrix_set(m, 5, 2, mu_r5 * (r2 - 3 * y[2] * y[2]))

    dfdt[0] = 0.0
    dfdt[1] = 0.0
    dfdt[2] = 0.0
    dfdt[3] = 0.0
    dfdt[4] = 0.0
    dfdt[5] = 0.0

    return cbsint.GSL_SUCCESS


def bsint(r, v, t, dt, beta):
    cdef double y[6]
    y[:] = (r[0], r[1], r[2], v[0], v[1], v[2])
    cdef double ti = t
    cdef double tf = t + dt
    cdef double params = beta

    cdef cbsint.gsl_odeiv2_system sys = {grav2_rp, jac_grav2_rp, 6, & params}

    cdef cbsint.gsl_odeiv2_driver * d = cbsint.gsl_odeiv2_driver_alloc_y_new(
        & sys, cbsint.gsl_odeiv2_step_bsimp, 1e-6, 1e-6, 0.0)

    cdef int status = cbsin.gsl_odeiv2_driver_apply(d, & ti, tf, y)

    if status != cbsint.GSL_SUCCESS:
        raise ValueError('Integration failed with code {}'.format(status))

    cbsint.gsl_odeiv2_driver_free(d)

    return (y[0], y[1], y[2]), (y[3], y[4], y[5])
