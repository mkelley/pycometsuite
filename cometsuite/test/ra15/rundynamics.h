/***************************************************************************
  Copyright (C) 2005-2010 by Michael S. Kelley <msk@astro.umd.edu>

  ***************************************************************************/

#if !defined(__RUNDYNAMICS)
#define __RUNDYNAMICS 1

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

/** The number of positions per planet to save in the lookup table.
    The maximum is somewhere below 2^31, but you will run into
    physical RAM limits before that.  2e6 needs about 200MB of RAM.
    In tests, 14 steps per day offers a 40% increase in performance
    but the error increases by a factor of 10. */
#define _N_LOOKUP_DATES 2000000

/** Keys to return extrema from a pfunction.  The age upper limit is
    necessary to create the planet lookup table. 

    /todo Convert these into an enumerated list.
*/
#define RADIUS_MAX 1
#define RADIUS_MIN 2
#define AGE_MAX 3
#define AGE_MIN 4
#define UNLIMITED -65536

/* Astronomical Constants */
#define _MU 1.32712440017987e11 /* (+/- 8) [km^3/s^2] (source: DE405) */
#define _C  299792.458       /* [km/s] */
#define _AU 149597870.691    /* [km] */

#define DEBUG 0

enum parseCommandLineFlags { CL_NOERROR, CL_HELP, CL_NOFILE, CL_VERSION,
                             CL_BADINPUT };

#endif
