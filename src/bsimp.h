/*
#include "gsl/gsl_math.h"
#include "gsl/gsl_errno.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_odeiv2.h"

#include "odeiv_util.h"
#include "step_utils.c"
*/



#ifndef __BSIMP_H__
#define __BSIMP_H__

/******************** gsl_machine *******************************/
#define GSL_SQRT_DBL_EPSILON   1.4901161193847656e-08
#define GSL_SQRT_DBL_MAX   1.3407807929942596e+154


/******************** gsl_block *******************************/
struct gsl_block_struct
{
  size_t size;
  double *data;
};

typedef struct gsl_block_struct gsl_block;

/******************** gsl_matrix *******************************/
typedef struct 
{
  size_t size1;
  size_t size2;
  size_t tda;
  double * data;
  gsl_block * block;
  int owner;
} gsl_matrix

gsl_matrix * 
gsl_matrix_alloc (const size_t n1, const size_t n2);
void gsl_matrix_free (gsl_matrix * m);

inline double   gsl_matrix_get(const gsl_matrix * m, const size_t i, const size_t j);
inline void     gsl_matrix_set(gsl_matrix * m, const size_t i, const size_t j, const double x);

inline
double
gsl_matrix_get(const gsl_matrix * m, const size_t i, const size_t j)
{
  return m->data[i * m->tda + j] ;
} 

inline
void
gsl_matrix_set(gsl_matrix * m, const size_t i, const size_t j, const double x)
{
  m->data[i * m->tda + j] = x ;
}

/******************** gsl_permutation *******************************/
struct gsl_permutation_struct
{
  size_t size;
  size_t *data;
};

typedef struct gsl_permutation_struct gsl_permutation;

gsl_permutation *gsl_permutation_alloc (const size_t n);
void gsl_permutation_free (gsl_permutation * p);

/******************** gsl_vector *******************************/
typedef struct 
{
  size_t size;
  size_t stride;
  double *data;
  gsl_block *block;
  int owner;
} 
gsl_vector;

typedef struct
{
  gsl_vector vector;
} _gsl_vector_view;

typedef _gsl_vector_view gsl_vector_view;

_gsl_vector_view 
gsl_vector_view_array (double *v, size_t n);


/******************** gsl_linalg *******************************/

int gsl_linalg_LU_decomp (gsl_matrix * A, gsl_permutation * p, int *signum);

int gsl_linalg_LU_solve (const gsl_matrix * LU,
                         const gsl_permutation * p,
                         const gsl_vector * b,
                         gsl_vector * x);

/******************** gsl_odeiv2 *******************************/

#define GSL_ODEIV_FN_EVAL(S,t,y,f)  (*((S)->function))(t,y,f,(S)->params)
#define GSL_ODEIV_JA_EVAL(S,t,y,dfdy,dfdt)  (*((S)->jacobian))(t,y,dfdy,dfdt,(S)->params)

typedef struct
{
  int (*function) (double t, const double y[], double dydt[], void *params);
  int (*jacobian) (double t, const double y[], double *dfdy, double dfdt[],
                   void *params);
  size_t dimension;
  void *params;
}
gsl_odeiv2_system;


/* Driver object
 *
 * This is a high level wrapper for step, control and
 * evolve objects. 
 */

struct gsl_odeiv2_driver_struct
{
  const gsl_odeiv2_system *sys; /* ODE system */
  gsl_odeiv2_step *s;           /* stepper object */
  gsl_odeiv2_control *c;        /* control object */
  gsl_odeiv2_evolve *e;         /* evolve object */
  double h;                     /* step size */
  double hmin;                  /* minimum step size allowed */
  double hmax;                  /* maximum step size allowed */
  unsigned long int n;          /* number of steps taken */
  unsigned long int nmax;       /* Maximum number of steps allowed */
};

typedef struct gsl_odeiv2_driver_struct gsl_odeiv2_driver;

/* Stepper object
 *
 * Opaque object for stepping an ODE system from t to t+h.
 * In general the object has some state which facilitates
 * iterating the stepping operation.
 */

typedef struct
{
  const char *name;
  int can_use_dydt_in;
  int gives_exact_dydt_out;
  void *(*alloc) (size_t dim);
  int (*apply) (void *state, size_t dim, double t, double h, double y[],
                double yerr[], const double dydt_in[], double dydt_out[],
                const gsl_odeiv2_system * dydt);
  int (*set_driver) (void *state, const gsl_odeiv2_driver * d);
  int (*reset) (void *state, size_t dim);
  unsigned int (*order) (void *state);
  void (*free) (void *state);
}
gsl_odeiv2_step_type;

#endif /* __BSIMP_H__ */
