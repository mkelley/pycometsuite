#include <stdio.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

/* Constsants from DE430/1:

  The Planetary and Lunar Ephemerides DE430 and DE431. Folkner et
  al. 2014. Interplanetary Network Progress Report 42-196

  Units: km, km/s, km3/s2

*/
#define AU 149597870.700
#define C 299792.458
#define MU 132712440041.939400

struct dust_parameters {
  double et, beta;
};

/* Dust orbiting the Sun with radiation pressure (no PR drag, no GR) */
int
dust (double t, const double y[], double f[], void *p)
{
  (void)(t); /* avoid unused parameter warning */
  
  struct dust_parameters * params = (struct dust_parameters *)p;
  double beta = (params->beta);

  double r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
  double r1 = sqrt(r2);
  double r3 = r1 * r2;
  double mu_r3 = MU / r3 * (1 - beta);

  f[0] = y[3];
  f[1] = y[4];
  f[2] = y[5];
  f[3] = -y[0] * mu_r3;
  f[4] = -y[1] * mu_r3;
  f[5] = -y[2] * mu_r3;

  return GSL_SUCCESS;
}

int
jac_dust (double t, const double y[], double *dfdy,
	  double dfdt[], void *p)
{
  (void)(t); /* avoid unused parameter warning */
    
  struct dust_parameters * params = (struct dust_parameters *)p;
  double beta = (params->beta);

  double r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
  double r1 = sqrt(r2);
  double r3 = r1 * r2;
  double mu_r5 = MU * (1 - beta) / r3 / r2;
  
  gsl_matrix_view dfdy_mat = gsl_matrix_view_array (dfdy, 6, 6);
  gsl_matrix * m = &dfdy_mat.matrix;

  gsl_matrix_set_zero(m);

  gsl_matrix_set (m, 0, 3, 1.0);
  gsl_matrix_set (m, 1, 4, 1.0);
  gsl_matrix_set (m, 2, 5, 1.0);

  gsl_matrix_set (m, 3, 0, mu_r5 * (r2 - 3 * y[0] * y[0]));
  gsl_matrix_set (m, 3, 1, -mu_r5 * 3 * y[0] * y[1]);
  gsl_matrix_set (m, 3, 2, -mu_r5 * 3 * y[0] * y[2]);
  
  gsl_matrix_set (m, 4, 0, -mu_r5 * 3 * y[1] * y[0]);
  gsl_matrix_set (m, 4, 1, mu_r5 * (r2 - 3 * y[1] * y[1]));
  gsl_matrix_set (m, 4, 2, -mu_r5 * 3 * y[1] * y[2]);
  
  gsl_matrix_set (m, 5, 0, -mu_r5 * 3 * y[2] * y[0]);
  gsl_matrix_set (m, 5, 1, -mu_r5 * 3 * y[2] * y[1]);
  gsl_matrix_set (m, 5, 2, mu_r5 * (r2 - 3 * y[2] * y[2]));

  dfdt[0] = 0.0;
  dfdt[1] = 0.0;
  dfdt[2] = 0.0;
  dfdt[3] = 0.0;
  dfdt[4] = 0.0;
  dfdt[5] = 0.0;

  return GSL_SUCCESS;
}

static int
bsint_dust (double init[], double ti, double dt, double beta,
	    double *final)
{
  struct dust_parameters p = { ti, beta };
  gsl_odeiv2_system sys = {dust, jac_dust, 6, &p};

  gsl_odeiv2_driver * d =
    gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_bsimp,
				   86400, 1e-6, 1e-6);
  double t = 0;
  double y[6] = { init[0], init[1], init[2], init[3], init[4], init[5] };

  int status = gsl_odeiv2_driver_apply (d, &t, dt, y);

  if (status != GSL_SUCCESS) {
    printf ("error, return value=%d\n", status);
  }

  final[0] = y[0];
  final[1] = y[1];
  final[2] = y[2];
  final[3] = y[3];
  final[4] = y[4];
  final[5] = y[5];

  gsl_odeiv2_driver_free (d);
  return 0;
}
