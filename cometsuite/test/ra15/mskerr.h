/***************************************************************************
  Copyright (C) 2004 Michael S. Kelley <msk@astro.umd.edu>
  
  ***************************************************************************/

#define _ERR(_s) if (_DEBUG) { fprintf(stderr, " %s", _s); }
#define _NL if (_DEBUG) { fprintf(stderr, "\n"); }
#define _I_ERR(_s, _i) if (_DEBUG) { fprintf(stderr, " %s: %d", _s, _i); }
#define _D_ERR(_s, _d) if (_DEBUG) { fprintf(stderr, " %s: %lf", _s, _d); }
#define _E_ERR(_s, _e) if (_DEBUG) { fprintf(stderr, " %s: %25.16e", _s, _e); }
#define _A_ERR(_s, _a, _n) if (_DEBUG) { for(i=0;i<_n;i++) fprintf(stderr, " %s[%d]: %25.16e\n", _s, i, _a[i]); }
#define _V_ERR(_s, _v) if (_DEBUG) { fprintf(stderr, " %s (x,y,z) = %25.16e, %25.16e, %25.16e", _s, _v[0], _v[1], _v[2]); }
