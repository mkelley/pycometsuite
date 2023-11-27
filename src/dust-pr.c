#include <stdio.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>

/*
gcc bs-example.c -lgsl -lgslcblas -lm
*/

/* km, km/s, km3/s2 */
#define AU 149597870.691
#define C 299792.458
#define MU 1.32712440017987e11

int
func (double t, const double y[], double f[],
	 void *params)
{
  (void)(t); /* avoid unused parameter warning */

  double r2 = y[0] * y[0] + y[1] * y[1] + y[2] + y[2];
  double r1 = sqrt(r2);
  double r3 = r1 * r2;
  double beta = *(double *)params;
  double mu_r3 = MU / r3;

  /* radiation force up to Poynting-Robertson drag */
  double[3] rhat = { y[0] / r1, y[1] / r1, y[2] / r1 };
  double vr_c = (y[3] * rhat[0] + y[4] * rhat[1] + y[5] * rhat[2]) / C;
  double betamu_r2 = beta * MU / r2;

  f[0] = y[3];
  f[1] = y[4];
  f[2] = y[5];
  f[3] = -y[0] * mu_r3 + betamu_r2 * ((1 - vr_c) * rhat[0] - y[3] / C);
  f[4] = -y[1] * mu_r3 + betamu_r2 * ((1 - vr_c) * rhat[1] - y[4] / C);
  f[5] = -y[2] * mu_r3 + betamu_r2 * ((1 - vr_c) * rhat[2] - y[5] / C);

  return GSL_SUCCESS;
}

int
jac (double t, const double y[], double *dfdy,
     double dfdt[], void *params)
{
  (void)(t); /* avoid unused parameter warning */
  
  double r2 = y[0] * y[0] + y[1] * y[1] + y[2] + y[2];
  double r1 = sqrt(r2);
  double r3 = r1 * r2;
  double beta = *(double *)params;
  double mu_r3 = MU * (1 - beta) / r3;
  double mu_r5 = mu_r3 / r2;
  
  gsl_matrix_view dfdy_mat
    = gsl_matrix_view_array (dfdy, 6, 6);
  gsl_matrix * m = &dfdy_mat.matrix;
  gsl_matrix_set (m, 0, 0, 0.0);
  gsl_matrix_set (m, 0, 1, 0.0);
  gsl_matrix_set (m, 0, 2, 0.0);
  gsl_matrix_set (m, 0, 3, 1.0);
  gsl_matrix_set (m, 0, 4, 0.0);
  gsl_matrix_set (m, 0, 5, 0.0);

  gsl_matrix_set (m, 1, 0, 0.0);
  gsl_matrix_set (m, 1, 1, 0.0);
  gsl_matrix_set (m, 1, 2, 0.0);
  gsl_matrix_set (m, 1, 3, 0.0);
  gsl_matrix_set (m, 1, 4, 1.0);
  gsl_matrix_set (m, 1, 5, 0.0);

  gsl_matrix_set (m, 2, 0, 0.0);
  gsl_matrix_set (m, 2, 1, 0.0);
  gsl_matrix_set (m, 2, 2, 0.0);
  gsl_matrix_set (m, 2, 3, 0.0);
  gsl_matrix_set (m, 2, 4, 0.0);
  gsl_matrix_set (m, 2, 5, 1.0);
  
  gsl_matrix_set (m, 3, 0, mu_r5 * (r2 - 3 * y[0] * y[0]));
  gsl_matrix_set (m, 3, 1, -mu_r5 * 3 * y[0] * y[1]);
  gsl_matrix_set (m, 3, 2, -mu_r5 * 3 * y[0] * y[2]);
  gsl_matrix_set (m, 3, 3, 0.0);
  gsl_matrix_set (m, 3, 4, 0.0);
  gsl_matrix_set (m, 3, 5, 0.0);
  
  gsl_matrix_set (m, 4, 0, -mu_r5 * 3 * y[1] * y[0]);
  gsl_matrix_set (m, 4, 1, mu_r5 * (r2 - 3 * y[1] * y[1]));
  gsl_matrix_set (m, 4, 2, -mu_r5 * 3 * y[1] * y[2]);
  gsl_matrix_set (m, 4, 3, 0.0);
  gsl_matrix_set (m, 4, 4, 0.0);
  gsl_matrix_set (m, 4, 5, 0.0);
  
  gsl_matrix_set (m, 5, 0, -mu_r5 * 3 * y[2] * y[0]);
  gsl_matrix_set (m, 5, 1, -mu_r5 * 3 * y[2] * y[1]);
  gsl_matrix_set (m, 5, 2, mu_r5 * (r2 - 3 * y[2] * y[2]));
  gsl_matrix_set (m, 5, 3, 0.0);
  gsl_matrix_set (m, 5, 4, 0.0);
  gsl_matrix_set (m, 5, 5, 0.0);

  dfdt[0] = 0.0;
  dfdt[1] = 0.0;
  dfdt[2] = 0.0;
  dfdt[3] = 0.0;
  dfdt[4] = 0.0;
  dfdt[5] = 0.0;

  return GSL_SUCCESS;
}

int
main (void)
{
  double beta = 0.45;
  gsl_odeiv2_system sys = {func, jac, 6, &beta};

  gsl_odeiv2_driver * d =
    gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_bsimp,
                                  1e-6, 1e-6, 0.0);
  int i;
  double t = 0.0, t1 = 365.25 * 86400 * 10;
  double y[6] = { 1 * AU, 0.0, 0.0, 0.0, 30.0, 0.0 };

  for (i = 1; i <= 1000; i++)
    {
      double ti = i * t1 / 1000.0;
      int status = gsl_odeiv2_driver_apply (d, &t, ti, y);

      if (status != GSL_SUCCESS)
        {
          printf ("error, return value=%d\n", status);
          break;
        }

      printf ("%.5e %.5e %.5e %.5e\n", t, y[0] / AU, y[1] / AU, y[2] / AU);
    }

  gsl_odeiv2_driver_free (d);
  return 0;
}
