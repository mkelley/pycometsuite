cdef extern from "gsl/gsl_errno.h":
    enum GSLErrors:
        GSL_SUCCESS,
        GSL_FAILURE,
        GSL_EUNDRFLW

cdef extern from "gsl/gsl_matrix.h":
    struct gsl_matrix:
        pass

    struct _gsl_matrix_view:
        gsl_matrix matrix

    _gsl_matrix_view gsl_matrix_view_array(
        double * base, const size_t n1, const size_t n2)

    void gsl_matrix_set_zero(gsl_matrix * m)

    inline void gsl_matrix_set(gsl_matrix * m, const size_t i,
                               const size_t j, const double x)

cdef extern from "gsl/gsl_odeiv2.h":
    struct gsl_odeiv2_system:
        pass

    struct gsl_odeiv2_driver:
        pass

    struct gsl_odeiv2_step_type:
        pass

    gsl_odeiv2_driver * gsl_odeiv2_driver_alloc_y_new(
        const gsl_odeiv2_system * sys,
        const gsl_odeiv2_step_type *
        T, const double hstart,
        const double epsabs,
        const double epsrel)

    int gsl_odeiv2_driver_apply(gsl_odeiv2_driver * d, double * t,
                                const double t1, double y[])

    void gsl_odeiv2_driver_free(gsl_odeiv2_driver * state)

    const gsl_odeiv2_step_type * gsl_odeiv2_step_bsimp
