/***************************************************************************

  Originally the RADU15.F integrator (Everhart 1985. in Dynamics of
  Comets: Their Origin and Evolution. A. Carusi and G. Valsecchi,
  eds. Astrophysics and Space Science Library 115 185).

 ***************************************************************************/

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <math.h>
#include <string.h>

#define true 1
#define false 0

#define _DEBUG 0
#define _ERR(_s) if (_DEBUG) { fprintf(stderr, " %s", _s); }
#define _NL if (_DEBUG) { fprintf(stderr, "\n"); }
#define _I_ERR(_s, _i) if (_DEBUG) { fprintf(stderr, " %s: %d", _s, _i); }
#define _D_ERR(_s, _d) if (_DEBUG) { fprintf(stderr, " %s: %lf", _s, _d); }
#define _E_ERR(_s, _e) if (_DEBUG) { fprintf(stderr, " %s: %25.16e", _s, _e); }
#define _A_ERR(_s, _a, _n) if (_DEBUG) { for(i=0;i<_n;i++) fprintf(stderr, " %s[%d]: %25.16e\n", _s, i, _a[i]); }
#define _V_ERR(_s, _v) if (_DEBUG) { fprintf(stderr, " %s (x,y,z) = %25.16e, %25.16e, %25.16e", _s, _v[0], _v[1], _v[2]); }

/* number of differential equations */
#define _NDE 3

/* Astronomical Constants */
#define _C  299792.458       /* [km/s] */
#define _AU 149597870.691    /* [km] */

/* temporary GM */
#define _MU 1.204e-8

/* Default initial step size. */
#define _ISS 1.0

/* h = t/tStep; fractional step sizes in terms of the total integration
   step size (Gaussian-Radau spacings sacled to the range [0,1] for
   integrating to order 15); the sum should be 3.7333333333333333 */
static const double h[] = {0.0, 0.05626256053692215, 0.18024069173689236,
			   0.35262471711316964, 0.54715362633055538,
			   0.73421017721541053, 0.88532094683909577,
			   0.97752061356128750};
static const double sr = 1.4;
static const double pw = 1.0 / 9.0;

int everhart(double *x, double *v, double t, double dt, double beta,
	     double tol, double *minStep, int *nFC, int *nSeq);
void initConstants(double *xc, double *vc, double *c, double *d, double *r);
int calcAccel(double *x, double *v, double t, double dt, double beta,
	      double *a);

/******************************************************************************/
/** Integrate the position of a particle under solar radiation and
    gravity forces.
*/
int everhart(double *x, double *v, double t, double dt, double beta,
	     double tol, double *minStep, int *nFC, int *nSeq) {
  static double c[21], d[21], r[21];
  static double xc[8], vc[7];
  double a1[_NDE], aj[_NDE], y[_NDE], z[_NDE];
  double b[7][_NDE], g[7][_NDE], e[7][_NDE], bd[7][_NDE];
  double s[9];
  double tPrime, tStep, tTotal;
  double xstep, dir, temp, gk, hv;
  static int initDone = false;
  int first_sequence, final_sequence, constant_step, redo_first_sequence;
  int i, j, k, l, m, nIter, nCount;

  if (_DEBUG) {
    _ERR("Input Variables:\n");
    _V_ERR("   x", x); _NL;
    _V_ERR("   v", v); _NL;
    _E_ERR("   dt", dt); _E_ERR(" t", t); _E_ERR(" beta", beta); _NL;
  }

  xstep = 1e3;

  /* zero some arrays */
  memset(b, 0, sizeof(double)*7*_NDE);
  memset(bd, 0, sizeof(double)*7*_NDE);
  memset(g, 0, sizeof(double)*7*_NDE);
  memset(e, 0, sizeof(double)*7*_NDE);

  if (!initDone) {
    initConstants(xc, vc, c, d, r);
    initDone = true;
  }

  /* Now that the constants are initialized, make an estimate of tPrime */
  dir = (dt<0.0)?-1.0:1.0;
  if (tol < 0.0) {
    constant_step = true;
    tPrime = xstep;
  } else {
    if (tol == 0.0) tol = 1e-8;
    constant_step = false;
    /*    tPrime = 0.1 * dir; */
    tPrime = _ISS * dir;
  }

  if ((tPrime / dt) > 0.5) tPrime = 0.5 * dt;

  nCount = 0;
  first_sequence = true;
  final_sequence = false;
  if (_DEBUG) printf("%15s %9s %24s %24s %12s %12s %24s\n",
		     "Function calls", "Sequences", "x[0]",
		     "v[0]", "tStep", "tTotal", "dt");

  do { /* while(1) */
    do { /* while(redo_first_sequence) */
      if (first_sequence) {
	*nSeq = 0;
	nIter = 6;
	tTotal = 0.0;
	*minStep = 0.0;
	if (calcAccel(x, v, 0.0, t, beta, a1))
	  return 1;
	*nFC = 1;
      }

      tStep = tPrime;
      if (*nSeq % 1000 == 0) {
	if (_DEBUG) printf("%15d %9d %24.16e %24.16e %12e %12e %24.16e\n",
			   *nFC, *nSeq, x[0], v[0], tStep, tTotal, dt);
      }

      /* Find new g values from predicted b values */
      for (k=0; k<_NDE; k++) {
	g[0][k] = d[15]*b[6][k] + d[10]*b[5][k] + d[6]*b[4][k] + d[3]*b[3][k] + d[1]*b[2][k] 
	  + d[0]*b[1][k] + b[0][k];
	g[1][k] = d[16]*b[6][k] + d[11]*b[5][k] + d[7]*b[4][k] + d[4]*b[3][k] + d[2]*b[2][k] +
	  b[1][k];
	g[2][k] = d[17]*b[6][k] + d[12]*b[5][k] + d[8]*b[4][k] + d[5]*b[3][k] +      b[2][k];
	g[3][k] = d[18]*b[6][k] + d[13]*b[5][k] + d[9]*b[4][k] +      b[3][k];
	g[4][k] = d[19]*b[6][k] + d[14]*b[5][k] +      b[4][k];
	g[5][k] = d[20]*b[6][k] +       b[5][k];
	g[6][k] =       b[6][k];
      }

      for (m=1; m<=nIter; m++) {
	for (j=1; j<8; j++) {
	  s[0] = tStep * h[j];
	  s[1] = s[0] * s[0];
	  s[2] = s[1] * h[j];         s[1] = s[1] * xc[0];
	  s[3] = s[2] * h[j];         s[2] = s[2] * xc[1];
	  s[4] = s[3] * h[j];         s[3] = s[3] * xc[2];
	  s[5] = s[4] * h[j];         s[4] = s[4] * xc[3];
	  s[6] = s[5] * h[j];         s[5] = s[5] * xc[4];
	  s[7] = s[6] * h[j];         s[6] = s[6] * xc[5];
	  s[8] = s[7] * h[j] * xc[7]; s[7] = s[7] * xc[6];

	  for (k=0; k<_NDE; k++) {
	    y[k] = x[k] + 
	         v[k] * s[0] +
	        a1[k] * s[1] +
	      b[0][k] * s[2] +
	      b[1][k] * s[3] +
	      b[2][k] * s[4] +
	      b[3][k] * s[5] +
	      b[4][k] * s[6] +
	      b[5][k] * s[7] +
	      b[6][k] * s[8];
	  }

	  s[0] = tStep * h[j];
	  s[1] = s[0] * h[j];
	  s[2] = s[1] * h[j];         s[1] = s[1] * vc[0];
	  s[3] = s[2] * h[j];         s[2] = s[2] * vc[1];
	  s[4] = s[3] * h[j];         s[3] = s[3] * vc[2];
	  s[5] = s[4] * h[j];         s[4] = s[4] * vc[3];
	  s[6] = s[5] * h[j];         s[5] = s[5] * vc[4];
	  s[7] = s[6] * h[j] * vc[6]; s[6] = s[6] * vc[5];
	  
	  for (k=0; k<_NDE; k++) {
	    z[k] = v[k] + 
	        a1[k] * s[0] +
	      b[0][k] * s[1] +
	      b[1][k] * s[2] +
	      b[2][k] * s[3] +
	      b[3][k] * s[4] +
	      b[4][k] * s[5] +
	      b[5][k] * s[6] +
	      b[6][k] * s[7];
	  }	  

	  if (calcAccel(y, z, tTotal + h[j] * tStep, t, beta, aj)) return 1;
	  (*nFC)++;

	  for (k=0; k<_NDE; k++) {
	    temp = g[j-1][k];
	    gk = (aj[k] - a1[k]) / h[j];

	    switch(j) {
	    case 1 : g[0][k] =       gk; break;
	    case 2 : g[1][k] =      (gk-g[0][k])*r[ 0]; break;
	    case 3 : g[2][k] =     ((gk-g[0][k])*r[ 1]-g[1][k])*r[ 2]; break;
	    case 4 : g[3][k] =    (((gk-g[0][k])*r[ 3]-g[1][k])*r[ 4]-g[2][k])*r[ 5]; break;
	    case 5 : g[4][k] =   ((((gk-g[0][k])*r[ 6]-g[1][k])*r[ 7]-g[2][k])*r[ 8]-g[3][k])*
		                 r[ 9]; break;
	    case 6 : g[5][k] =  (((((gk-g[0][k])*r[10]-g[1][k])*r[11]-g[2][k])*r[12]-g[3][k])*
				 r[13]-g[4][k])*r[14]; break;
	    case 7 : g[6][k] = ((((((gk-g[0][k])*r[15]-g[1][k])*r[16]-g[2][k])*r[17]-g[3][k])*
				 r[18]-g[4][k])*r[19]-g[5][k])*r[20]; break;
	    }

	    temp = g[j-1][k] - temp;
	    b[j-1][k] = b[j-1][k] + temp;

	    switch(j) {
	    case 2 : b[0][k] = b[0][k] + c[0] * temp;
	             break;
	    case 3 : b[0][k] = b[0][k] + c[1] * temp;
	             b[1][k] = b[1][k] + c[2] * temp;
		     break;
	    case 4 : b[0][k] = b[0][k] + c[3] * temp;
	             b[1][k] = b[1][k] + c[4] * temp;
	             b[2][k] = b[2][k] + c[5] * temp;
		     break;
	    case 5 : b[0][k] = b[0][k] + c[6] * temp;
	             b[1][k] = b[1][k] + c[7] * temp;
	             b[2][k] = b[2][k] + c[8] * temp;
	             b[3][k] = b[3][k] + c[9] * temp;
		     break;
	    case 6 : b[0][k] = b[0][k] + c[10] * temp;
	             b[1][k] = b[1][k] + c[11] * temp;
	             b[2][k] = b[2][k] + c[12] * temp;
	             b[3][k] = b[3][k] + c[13] * temp;
	             b[4][k] = b[4][k] + c[14] * temp;
		     break;
	    case 7 : b[0][k] = b[0][k] + c[15] * temp;
	             b[1][k] = b[1][k] + c[16] * temp;
	             b[2][k] = b[2][k] + c[17] * temp;
	             b[3][k] = b[3][k] + c[18] * temp;
	             b[4][k] = b[4][k] + c[19] * temp;
	             b[5][k] = b[5][k] + c[20] * temp;
		     break;
	    }
	  } /* End k loop */
	} /* End j loop */
      } /* End nIter loop */

      if (!constant_step) {
	/* Sequence size control */
	hv = 0.0;
	for (k=0; k<_NDE; k++) {
	  hv = (hv>fabs(b[6][k]))?hv:fabs(b[6][k]);
	}
	hv *= xc[7] / pow(fabs(tStep), 7);
      }

      redo_first_sequence = false;
      if (first_sequence) {
	if (constant_step) {
	  tPrime = xstep;
	} else {
	  tPrime = dir * pow(tol / hv, pw);
	  if ((tPrime / tStep) <= 1.0) {
	    /* If the new tPrime is smaller than the last tStep */
	    /* then restart with: */
	    tPrime = 0.8 * tPrime;
	    nCount++;
	    if (nCount > 1) if (_DEBUG) printf("%15d %24.16e %24.16e\n", nCount,tStep,tPrime);

	    if (nCount > 10) {
	      fprintf(stderr, "Too many time refinements!\n");
	      return 1;
	    }
	  }
	}
	if (!redo_first_sequence) first_sequence = false;
      }
    } while(redo_first_sequence);

    /* Find the new x and v values */
    s[0] = tStep;
    s[1] = s[0] * s[0];
    s[2] = s[1] * xc[1];
    s[3] = s[1] * xc[2];
    s[4] = s[1] * xc[3];
    s[5] = s[1] * xc[4];
    s[6] = s[1] * xc[5];
    s[7] = s[1] * xc[6];
    s[8] = s[1] * xc[7];
    s[1] = s[1] * xc[0];

    for (k=0; k<_NDE; k++) {
      x[k] = x[k] + 
	   v[k] * s[0] +
	  a1[k] * s[1] +
	b[0][k] * s[2] +
	b[1][k] * s[3] +
	b[2][k] * s[4] +
	b[3][k] * s[5] +
	b[4][k] * s[6] +
	b[5][k] * s[7] +
	b[6][k] * s[8];
    }
      
    s[0] = tStep;
    s[1] = s[0] * vc[0];
    s[2] = s[0] * vc[1];
    s[3] = s[0] * vc[2];
    s[4] = s[0] * vc[3];
    s[5] = s[0] * vc[4];
    s[6] = s[0] * vc[5];
    s[7] = s[0] * vc[6];
  
    for (k=0; k<_NDE; k++) {
      v[k] = v[k] + 
	a1[k] * s[0] +
	b[0][k] * s[1] +
	b[1][k] * s[2] +
	b[2][k] * s[3] +
	b[3][k] * s[4] +
	b[4][k] * s[5] +
	b[5][k] * s[6] +
	b[6][k] * s[7];
    }

    tTotal += tStep;
    if ((*minStep > tStep) || (*minStep == 0.0)) *minStep = tStep;
    (*nSeq)++;

    /* If we are done, then return */
    if (final_sequence) {
      if (_DEBUG) printf("%15d %9d %24.16e %24.16e %12e %12e %24.16e (final)\n",
			 *nFC, *nSeq, x[0], v[0], tStep, tTotal, dt);
      if (_DEBUG) printf("adsf");
      if (_DEBUG) printf("adsf");
      if (_DEBUG) printf("adsf");
      return 0;
    }

    /* Control the size of the next sequence and adjust the last sequence */
    /* to exactly cover the integration span. */
    if (constant_step) {
      tPrime = xstep;
    } else {
      tPrime = dir * pow(tol / hv, pw);
      if ((tPrime / tStep) > sr) tPrime = tStep * sr;
    }

    if ((dir * (tTotal + tPrime)) >= (dir * dt - 1e-8)) {
      tPrime = dt - tTotal;
      final_sequence = true;
    }

    /* Get the acceleration at the begining of the next sequence */
    if (calcAccel(x, v, tTotal, t, beta, a1))
      return 1;
    (*nFC)++;

    /* Predict b values for the next step.  Values from the preceeding */
    /* sequence were saved in the e matrix.  The correction bd is
       applied below */
    s[0] = tPrime / tStep;
    s[1] = s[0] * s[0];
    s[2] = s[1] * s[0];
    s[3] = s[2] * s[0];
    s[4] = s[3] * s[0];
    s[5] = s[4] * s[0];
    s[6] = s[5] * s[0];

    for (k=0; k<_NDE; k++) {
      if (*nSeq != 1) {
	for (j=0; j<7; j++) {
	  bd[j][k] = b[j][k] - e[j][k];
	}
      }

      e[0][k] = s[0]*( 7.0*b[6][k] +  6.0*b[5][k] +  5.0*b[4][k] + 4.0*b[3][k] + 3.0*b[2][k] + 
		       2.0*b[1][k] + b[0][k]);
      e[1][k] = s[1]*(21.0*b[6][k] + 15.0*b[5][k] + 10.0*b[4][k] + 6.0*b[3][k] + 3.0*b[2][k] +
		      b[1][k]);
      e[2][k] = s[2]*(35.0*b[6][k] + 20.0*b[5][k] + 10.0*b[4][k] + 4.0*b[3][k] +     b[2][k]);
      e[3][k] = s[3]*(35.0*b[6][k] + 15.0*b[5][k] +  5.0*b[4][k] +     b[3][k]);
      e[4][k] = s[4]*(21.0*b[6][k] +  6.0*b[5][k] +      b[4][k]);
      e[5][k] = s[5]*( 7.0*b[6][k] +      b[5][k]);
      e[6][k] = s[6]*(     b[6][k]);

      for (l=0; l<7; l++) b[l][k] = e[l][k] + bd[l][k];
    }

    /* Two iterations for every sequence */
    nIter = 2;
  } while(1);

  return 0;
}

/******************************************************************************/
/* initConstants()
   One time initialization of constants.
*/
void initConstants(double *xc, double *vc, double *c, double *d, double *r) {
  int k, l, la, lb, lc, ld, le;
  int nw[] = {-1, -1, 0, 2, 5, 9, 14, 20};

  memset(c, 0, sizeof(double)*21);
  memset(d, 0, sizeof(double)*21);
  memset(r, 0, sizeof(double)*21);

  /* Prepare the constants */
  xc[0] = 1.0 / 2.0;
  xc[1] = 1.0 / 6.0;
  xc[2] = 1.0 / 12.0;
  xc[3] = 1.0 / 20.0;
  xc[4] = 1.0 / 30.0;
  xc[5] = 1.0 / 42.0;
  xc[6] = 1.0 / 56.0;
  xc[7] = 1.0 / 72.0;

  vc[0] = 1.0 / 2.0;
  vc[1] = 1.0 / 3.0;
  vc[2] = 1.0 / 4.0;
  vc[3] = 1.0 / 5.0;
  vc[4] = 1.0 / 6.0;
  vc[5] = 1.0 / 7.0;
  vc[6] = 1.0 / 8.0;

  c[0] = -h[1];
  d[0] = h[1];
  r[0] = 1.0 / (h[2] - h[1]); 
 la = 0;
  lc = 0;

  for (k=2; k<7; k++) {
    lb = la;
    la = lc + 1;
    lc = nw[k+1];

    c[la] = -h[k] * c[lb];
    c[lc] = c[la-1] - h[k];
    d[la] = h[1] * d[lb];
    d[lc] = -c[lc];
    r[la] = 1.0 / (h[k+1] - h[1]);
    r[lc] = 1.0 / (h[k+1] - h[k]);
    if (k > 2) {
      for (l=2; l<k; l++) {
	ld = la + l - 1;
	le = lb + l - 2;
	c[ld] = c[le] - h[k] * c[le+1];
	d[ld] = d[le] + h[l] * d[le+1];
	r[ld] = 1.0 / (h[k+1] - h[l]);
      }
    }
  }
  return;
}

/******************************************************************************/
/* calcAccel()
*/
int calcAccel(double *R, double *v, double t, double dt, double beta,
	      double *a) {
  double R1, R2, R3;
  double now;

  /* distance and (the same)^2 and ^3 */
  R1 = sqrt(R[0]*R[0] + R[1]*R[1] + R[2]*R[2]);
  R2 = pow(R1, 2);
  R3 = pow(R1, 3);

  /* gravitational acceleration */
  a[0] = -_MU * R[0] / R3;
  a[1] = -_MU * R[1] / R3;
  a[2] = -_MU * R[2] / R3;

  /* Add radiation forces (including Poynting-Robertson drag) */
  if (beta != 0.0) {
    rhat[0] = R[0] / R1;
    rhat[1] = R[1] / R1;
    rhat[2] = R[2] / R1;

    /* v dot rhat = radial velocity */
    vr = v[0]*rhat[0] + v[1]*rhat[1] + v[2]*rhat[2];

    a[0] += beta * _MU / R2 * ((1.0 - vr / _C) * rhat[0] - v[0] / _C);
    a[1] += beta * _MU / R2 * ((1.0 - vr / _C) * rhat[1] - v[1] / _C);
    a[2] += beta * _MU / R2 * ((1.0 - vr / _C) * rhat[2] - v[2] / _C);
  }

  now = t + dt;
  return 0;
}
