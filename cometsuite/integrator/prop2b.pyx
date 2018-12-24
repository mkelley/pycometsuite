cimport cython
import numpy as np
from libc.math cimport sqrt, fabs, log, exp, cos, sin, cosh, sinh
from libc.float cimport DBL_MAX


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def prop2b(double GM, state, double dt):
    """Propagate a state vector.

    Based on Universal Variables approach in CSPICE toolkit.

    """
    cdef int lcount, mostc
    cdef double r0, rv, h2, e, q, f, b, br0, bq, b2rv, q_r0
    cdef double maxc, logmxc, logdpm, fixed, rootf, logf, bound
    cdef double x, fx2, c0, c1, c2, c3, kfun, upper, lower, oldx
    cdef double kfunl, kfunu, x2, x3, br, pc, vc, pcdot, vcdot
    cdef double[3] r, v, h, eq

    r[0] = state[0]
    r[1] = state[1]
    r[2] = state[2]
    v[0] = state[3]
    v[1] = state[4]
    v[2] = state[5]

    r0 = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
    rv = np.dot(r, v)

    # specific angular momentum
    h = np.cross(r, v)
    h2 = np.dot(h, h)

    # eccentricity and periapsis vector (q)
    eq = np.cross(v, h) / GM - np.array(r) / r0
    e = np.sqrt(np.dot(eq, eq))
    q = h2 / (GM * (e + 1))
    print(GM, r, v, e)

    # constants
    f = 1 - e
    b = np.sqrt(q / GM)
    br0 = b * r0
    b2rv = b * b * rv
    bq = b * q
    q_r0 = q / r0

    # computing maximum of Stumpff coefficients
    maxc = np.max((1.0, fabs(br0), fabs(b2rv), fabs(bq), fabs(q_r0 / bq)))

    if f < 0:
        logmxc = log(maxc)
        logdpm = log(DBL_MAX / 2)
        fixed = logdpm - logmxc
        rootf = sqrt(-f)
        logf = log(-f)
        bound = min(fixed / rootf, (fixed + logf * 1.5) / rootf)
    else:
        bound = exp((log(1.5) + log(DBL_MAX) - log(maxc)) / 3)

    x = bracket(dt / bq, -bound, bound)
    fx2 = f * x * x
    c0, c1, c2, c3 = stmp03(fx2)
    kfun = x * (br0 * c1 + x * (b2rv * c2 + x * bq * c3))

    if dt < 0:
        upper = 0
        lower = x
        while kfun > dt:
            upper = lower
            lower *= 2
            oldx = x
            x = bracket(lower, -bound, bound)
            if x == oldx:
                fx2 = f * bound * bound
                c0, c1, c2, c3 = stmp03(fx2)
                kfunl = -bound * (br0 * c1 - bound *
                                  (b2rv * c2 - bound * bq * c3))
                kfunu = bound * (br0 * c1 + bound *
                                 (b2rv * c2 + bound * bq * c3))
                raise ValueError(
                    ('dt ({}) is beyond the range for which we can '
                     'reliably propagate states.  The limits for this '
                     'GM and initial state are from {} to {}')
                    .format(dt, kfunl, kfunu))
            fx2 = f * x * x
            c0, c1, c2, c3 = stmp03(fx2)
            kfun = x * (br0 * c1 + x * (b2rv * c2 + x * bq * c3))
    elif dt > 0:
        lower = 0
        upper = x
        while kfun < dt:
            lower = upper
            upper *= 2
            oldx = x
            x = bracket(upper, -bound, bound)
            if x == oldx:
                fx2 = f * bound * bound
                c0, c1, c2, c3 = stmp03(fx2)
                kfunl = -bound * (br0 * c1 - bound *
                                  (b2rv * c2 - bound * bq * c3))
                kfunu = bound * (br0 * c1 + bound *
                                 (b2rv * c2 + bound * bq * c3))
                raise ValueError(
                    ('dt ({}) is beyond the range for which we can '
                     'reliably propagate states.  The limits for this '
                     'GM and initial state are from {} to {}')
                    .format(dt, kfunl, kfunu))
            fx2 = f * x * x
            c0, c1, c2, c3 = stmp03(fx2)
            kfun = x * (br0 * c1 + x * (b2rv * c2 + x * bq * c3))
    else:
        # dt == 0, nothing to do
        return state

    x = min(upper, max(lower, (lower + upper) / 2))
    fx2 = f * x * x
    c0, c1, c2, c3 = stmp03(fx2)
    lcount = 0
    mostc = 1000
    while (x > lower) and (x < upper) and (lcount < mostc):
        kfun = x * (br0 * c1 + x * (b2rv * c2 + x * bq * c3))
        if kfun > dt:
            upper = x
        elif kfun < dt:
            lower = x
        else:
            upper = x
            lower = x

        # set limits once upper and lower != 0
        if mostc > 64:
            if (upper != 0) and (lower != 0):
                mostc = 64
                lcount = 0

        x = min(upper, max(lower, (lower + upper) / 2))
        fx2 = f * x * x
        c0, c1, c2, c3 = stmp03(fx2)
        lcount += 1

    x2 = x * x
    x3 = x2 * x
    br = br0 * c0 + x * (b2rv * c1 + x * (bq * c2))
    pc = 1 - q_r0 * x2 * c2
    vc = dt - bq * x3 * c3
    pcdot = -q_r0 / br * x * c1
    vcdot = 1 - bq / br * x2 * c2

    state_f = np.empty(6)
    state_f[0] = pc * r[0] + vc * v[0]
    state_f[1] = pc * r[1] + vc * v[1]
    state_f[2] = pc * r[2] + vc * v[2]
    state_f[3] = pcdot * r[0] + vcdot * v[0]
    state_f[4] = pcdot * r[1] + vcdot * v[1]
    state_f[5] = pcdot * r[2] + vcdot * v[2]
    return state_f


cdef double bracket(double v, double mn, double mx):
    if v < mn:
        return mn
    elif v > mx:
        return mx
    else:
        return v


cdef double[20] PAIRS
cdef double LBOUND
for i in range(20):
    PAIRS[i] = 1 / (< double > (i + 1) * <double > (i + 2))

LBOUND = -(log(2) + log(DBL_MAX))**2


def stmp03(double x):
    cdef double z, c0, c1, c2, c3

    if x <= LBOUND:
        raise ValueError('x too small')

    if x < -1:
        z = sqrt(-x)
        c0 = cosh(z)
        c1 = sinh(z) / z
        c2 = (1 - c0) / x
        c3 = (1 - c1) / x
    elif x > 1:
        z = sqrt(x)
        c0 = cos(z)
        c1 = sin(z) / z
        c2 = (1 - c0) / x
        c3 = (1 - c1) / x
    else:
        c3 = 1
        for i in range(17, 1, -2):
            c3 = 1 - x * PAIRS[i] * c3
        c3 = PAIRS[1] * c3

        c2 = 1
        for i in range(16, 0, -2):
            c2 = 1 - x * PAIRS[i] * c2
        c2 = PAIRS[0] * c2

        c1 = 1 - x * c3
        c0 = 1 - x * c2

    return c0, c1, c2, c3
