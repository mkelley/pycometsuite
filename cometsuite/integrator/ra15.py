"""RADAU15 from Everhart 1985.

Dynamics of Comets: Their Origin and Evolution. A. Carusi and
G. Valsecchi, eds. Astrophysics and Space Science Library 115 185

Gauchez et al. (2002, in Modern Celestial Mechanics: From Theory to
Applications) demonstrate that RADAU15 is efficient and accurate up to
e=0.95.  Beyond that, the error increases because RADAU15 just guesses
what the step size should be but does not verify it.  RKN12 and
Bulirsch-Stoer work more generally and should be considered for
e>0.95.  However, Gauchez was testing 1000 orbits, and I cannot see
their plot in the Google Books preview.  I will need to trace down
this reference or some similar one.

.. todo: Write in a compiled language.

"""

import numpy as np
from .core import Integrator

class Ra15(Integrator):
    """RADAU15 from Everhart 1985.

    Parameters
    ----------
    iss : float, optional
      Initial step size. [s]
    tol : float, optional
      Error tolerance on iterations for determining next variable step
      size.  Set to `<0` to use `iss` as a constant step size.
    planets : list, optional
      Set to a list of NAIF IDs for perturbing planet systems to
      consider, e.g., `range(1, 10)`.

    """

    C = 299792.458  # km/s
    AU = 149597870.691  # km
    μ_sun = 1.32712440017987e11  # (+/- 8) km^3/s^2 (source: DE405)
    # GM for the planets
    μ_planets = (22032.080486417923,      # mer
                 324858.598826459784,     # ven
                 403503.233479087008,     # ear+moon
                 42828.314258067236,      # mars sys
                 126712767.857795968652,  # jup sys
                 37940626.061137281358,   # sat sys
                 5794549.007071875036,    # ura sys
                 6836534.063971338794,    # nep sys
                 981.600887707004 )       # plu sys

    # planet-comet distance required to be considered a close approach
    ca_dist = (0.1 * AU, 0.1 * AU, 0.1 * AU,
               0.1 * AU, 1.0 * AU, 1.0 * AU,
               1.0 * AU, 1.0 * AU, 0.1 * AU)

    # the semi-major axis of each planet
    a_planet = (0.387 * AU, 0.723 * AU,
                1.000 * AU, 1.523 * AU,
                5.203 * AU, 9.537 * AU,
                19.191 * AU, 30.068 * AU,
                39.481 * AU);

    def __init__(self, iss=86400, tol=1e-8,
                 planets=[], planet_lookup=False,
                 close_approaches=True, debug=False):
        self.iss = iss
        self.tol = tol
        self.planets = planets
        self.planet_lookup = planet_lookup
        self.close_approaches = close_approaches
        self._debug = debug

    @property
    def planets(self):
        return self._planets

    @planets.setter
    def planets(self, p):
        self._planets = tuple(sorted(p))
        if len(self._planets) > 0:
            self._perturbations = True
            self._n_planets = len(self._planets)
            self._μ_planets = np.array([self.μ_planets[i - 1]
                                        for i in self._planets])
        else:
            self._perturbations = False
            self._μ_planets = tuple()

    def debug(self, *args, end='\n'):
        import sys
        if self._debug:
            print(*args, file=sys.stderr, flush=True, end=end)

    def integrate(self, init, dt, beta=0):
        from ..state import State
        if dt == 0:
            return init
        r, v = self._ra15(init.r, init.v, beta, dt, init.t)
        return State(r, v, init.t + dt / 86400)

    def _ra15(self, x, v, beta, dt, et):
        # h = t / t_step; fractional step sizes in terms of the total
        # integration step size (Gaussian-Radau spacings sacled to the
        # range [0,1] for integrating to order 15); the sum should be
        # 3.7333333333333333
        h = (0.0, 0.05626256053692215, 0.18024069173689236,
             0.35262471711316964, 0.54715362633055538,
             0.73421017721541053, 0.88532094683909577,
             0.97752061356128750)

        emat = np.matrix([[1, 2, 3, 4,  5,  6,  7],
                          [0, 1, 3, 6, 10, 15, 21],
                          [0, 0, 1, 4, 10, 20, 35],
                          [0, 0, 0, 1,  5, 15, 35],
                          [0, 0, 0, 0,  1,  6, 21],
                          [0, 0, 0, 0,  0,  1,  7],
                          [0, 0, 0, 0,  0,  0,  1]])

        b, bd, g, e = np.zeros((4, 3, 7))
        sx = np.zeros(9)
        sv = np.zeros(8)

        # initialize constants
        xc = 1.0 / np.array([2., 6, 12, 20, 30, 42, 56, 72])
        vc = 1.0 / np.array([2., 3, 4, 5, 6, 7, 8])

        c, d, r = np.zeros((3, 21))
        c[0] = -h[1]
        d[0] = h[1]
        r[0] = 1 / (h[2] - h[1])
        la = 0
        lc = 0

        nw = (-1, -1, 0, 2, 5, 9, 14, 20)
        for k in range(2, 7):
            lb = la
            la = lc + 1;
            lc = nw[k+1]

            c[la] = -h[k] * c[lb]
            c[lc] = c[la-1] - h[k]
            d[la] = h[1] * d[lb]
            d[lc] = -c[lc]
            r[la] = 1 / (h[k+1] - h[1])
            r[lc] = 1 / (h[k+1] - h[k])
            if k > 2:
                self.debug('k > 2')
                for l in range(2, k):
                    ld = la + l - 1;
                    le = lb + l - 2;
                    c[ld] = c[le] - h[k] * c[le+1];
                    d[ld] = d[le] + h[l] * d[le+1];
                    r[ld] = 1 / (h[k+1] - h[l]);

        # estimate t_prime
        direction = np.sign(dt).astype(int)
        tol = self.tol
        if tol < 0.0:
            self.debug('tol < 0.0')
            constant_step = True;
            t_prime = direction * self.iss
        else:
            if tol == 0:
                self.debug('tol == 0')
                tol = 1e-8
            constant_step = False
            t_prime = direction * self.iss;

        if (t_prime / dt) > 0.5:
            self.debug('tprime/dt>0.5')            
            t_prime = 0.5 * dt

        self.n_fcalls = 0
        self.n_sequences = 0
        n_iterations = 6
        restarts = 0
        first_sequence = True
        final_sequence = False
        
        while True:
            self.debug('\n\n***** start *****')
            if first_sequence:
                self.debug('first sequence (start)')
                self.n_sequences = 0
                n_iterations = 6
                t_total = 0
                min_step = 0
                try:
                    a1 = self.accel(x, v, beta, 0.0, et)
                except:
                    raise

            t_step = t_prime

            # find new g values from predicted b values
            # .. todo: can this be rewritten to be more Pythonic?
            for k in (0, 1, 2):
                g[k][0] = d[15]*b[k][6] + d[10]*b[k][5] + d[6]*b[k][4] + d[3]*b[k][3] + d[1]*b[k][2] + d[0]*b[k][1] + b[k][0]
                g[k][1] = d[16]*b[k][6] + d[11]*b[k][5] + d[7]*b[k][4] + d[4]*b[k][3] + d[2]*b[k][2] + b[k][1]
                g[k][2] = d[17]*b[k][6] + d[12]*b[k][5] + d[8]*b[k][4] + d[5]*b[k][3] +      b[k][2]
                g[k][3] = d[18]*b[k][6] + d[13]*b[k][5] + d[9]*b[k][4] +      b[k][3]
                g[k][4] = d[19]*b[k][6] + d[14]*b[k][5] +      b[k][4]
                g[k][5] = d[20]*b[k][6] +       b[k][5]
                g[k][6] =       b[k][6]

            for m in range(n_iterations):
                ci = 0
                ri = 0
                for j in range(1, 8):
                    # position predictors
                    sx[0] = t_step * h[j]
                    sx[1] = sx[0]**2
                    sx[2:] = sx[1] * h[j]**np.arange(1, 8)
                    sx[1:] = sx[1:] * xc

                    y = x + v * sx[0] + a1 * sx[1] + np.sum(b * sx[2:], 1)

                    # velocity predictors
                    sv = t_step * h[j]**(1 + np.arange(8))
                    sv[1:] = sv[1:] * vc

                    z = v + a1 * sv[0] + np.sum(b * sv[1:], 1)

                    try:
                        aj = self.accel(y, z, beta, t_total + h[j] + t_step, et)
                    except:
                        raise

                    temp = g[:, j - 1].copy()
                    gk = (aj - a1) / h[j]

                    #                        if j == 1:
                    #                            g[:, 0] = gk
                    #                        else:
                    #                            rr = np.cumprod(r[ri:ri + j - 1][::-1])[::-1]
                    #                            g[:, j - 1] = gk * rr[0] - np.sum(gk * rr, 1)
                    #                            ri += j - 1

                    if j == 1:
                        g[:, 0] =       gk
                    elif j == 2:
                        g[:, 1] =      (gk - g[:, 0]) *  r[0]
                    elif j == 3:
                        g[:, 2] =     ((gk - g[:, 0]) *  r[1] - g[:, 1]) *  r[2]
                    elif j == 4:
                        g[:, 3] =    (((gk - g[:, 0]) *  r[3] - g[:, 1]) *  r[4] - g[:, 2]) *  r[5]
                    elif j == 5:
                        g[:, 4] =   ((((gk - g[:, 0]) *  r[6] - g[:, 1]) *  r[7] - g[:, 2]) *  r[8] - g[:, 3]) *  r[9]
                    elif j == 6:
                        g[:, 5] =  (((((gk - g[:, 0]) * r[10] - g[:, 1]) * r[11] - g[:, 2]) * r[12] - g[:, 3]) * r[13] - g[:, 4]) * r[14]
                    elif j == 7:
                        g[:, 6] = ((((((gk - g[:, 0]) * r[15] - g[:, 1]) * r[16] - g[:, 2]) * r[17] - g[:, 3]) * r[18] - g[:, 4]) * r[19] - g[:, 5]) * r[20]

                    temp = g[:, j - 1] - temp
                    b[:, j - 1] += temp

                    for i in range(2, j + 1):
                        b[:, i - 2] += c[ci] * temp
                        ci += 1
                # for j
            # for m  (n_iterations)

            if not constant_step:
                self.debug('not constant step (hv)')
                hv = np.fabs(b[:, 6]).max() * xc[7] / np.fabs(t_step)**7 

            if first_sequence:
                self.debug('first sequence step control')
                if constant_step:
                    self.debug('constant step')
                    t_prime = direction * self.iss
                else:
                    self.debug('not constant step')
                    t_prime = direction * (tol / hv)**(1/9)
                    if t_prime / t_step <= 1.0:
                        self.debug('tprime/tstep <= 1.0', t_prime, t_step)
                        # restart with a smaller t_prime
                        t_prime *= 0.8
                        restarts += 1
                        if restarts > 10:
                            raise ValueError("Too many time refinements on initial sequence.  Reduce initial step size.")
                        continue

                first_sequence = False

            # Executed on all sequences:
            # new x and v
            sx[0] = t_step
            sx[1:] = t_step**2 * xc
            x = x + v * sx[0] + a1 * sx[1] + np.sum(b * sx[2:], 1)

            sv[0] = t_step
            sv[1:] = t_step * vc
            v = v + a1 * sv[0] + np.sum(b * sv[1:], 1)

            t_total += t_step
            if (min_step > direction * t_step) or (min_step == 0):
                self.debug('minstep>tstep or minstep==0')
                min_step = t_step

            self.n_sequences += 1

            if final_sequence:
                return x, v

            # control the size of the next sequence
            if constant_step:
                self.debug('constant step (size control)')
                t_prime = direction * self.iss
            else:
                self.debug('not constant step (size control)')
                t_prime = direction * (tol / hv)**(1/9)
                if (np.abs(t_prime / t_step)) > 1.4:
                    self.debug('tprime/tstep > 1.4')
                    t_prime = t_step * 1.4

            if np.abs(t_total + t_prime) >= np.abs(dt - 1e-8):
                self.debug('ttotal+tprime > dt (set final sequence)')
                t_prime = dt - t_total
                final_sequence = True

            try:
                a1 = self.accel(x, v, beta, t_total, et)
            except:
                raise

            # Predict b values for next step.  Values from preceeding
            # sequence were saved in e matrix.  The correction bd is
            # applied below.
            s = (t_prime / t_step)**np.arange(1, 8)

            if self.n_sequences != 1:
                self.debug('nseq != 1')
                bd = b - e

            e = s * ((emat * b.T).A).T
            b = e + bd

            # two iterations for every sequence from now on
            n_iter = 2
        # end while True

    def accel(self, r, v, beta, dt, et):
        import spiceypy.wrapper as spice

        self.n_fcalls += 1

        r2 = np.dot(r, r)
        r1 = np.sqrt(r2)
        r3 = r1 * r2
        r_hat = r / r1

        # g_sun
        a = -self.μ_sun * r / r3

        # radiation force (up to Poyning-Robertson drag)
        if beta != 0.0:
            v_r = np.dot(v, r_hat)
            a += beta * self.μ_sun / r2 * ((1 - v_r / self.C) * r_hat - v / self.C)

        if self._perturbations:
            now = et + dt

            if self.planet_lookup:
                # planet lookup code here
                pass
            else:
                r_planet = np.empty((3, self._n_planets))
                ssb_sun = spice.spkssb(10, now, "ECLIPJ2000")[:3]  # wrt solar system barycenter
                for i in range(self._n_planets):
                    #state, ltt = spice.spkez(self._planets[i], now, "ECLIPJ2000", "NONE", 10)
                    r_planet[:, i] = spice.spkssb(self._planets[i], now, "ECLIPJ2000")[:3] - ssb_sun
                    #r_planet[:, i] = np.array([ -7.90761330e+08 - i * 1000,   1.82191521e+08 + i,   1.69376794e+07 + i]) - ssb_sun

            r_planet3 = np.sum(r_planet**2, 0)**(3/2)
            r_pp = (r_planet.T - r).T  # planet particle vector
            r_pp3 = np.sum(r_pp**2, 0)**(3/2)

            # first term is the direct acceleration, second is the
            # inderect accel because using the Sun as the origin is
            # not the center of mass of the system

            a += np.sum(self.μ_planets * (r_pp / r_pp3 - r_planet / r_planet3), 1)

        return a

    integrate.__doc__ = Integrator.integrate.__doc__

