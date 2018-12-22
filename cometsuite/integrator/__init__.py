"""

Integrate particle motions.

"""

__all__ = [
    'Kepler',
    'Ra15'
]

from . import core
from . import kepler
from . import ra15

from .kepler import Kepler
from .ra15 import Ra15

def test():
    import numpy as np
    import astropy.units as u
    import mskpy
    from mskpy.ephem.core import time2et
    from ..state import State

    dates = ['2016-02-22', '2017-02-22']
    ti = mskpy.cal2time(dates[0])
    ri, vi = mskpy.getxyz('encke', ti)
    ti_et = time2et(ti)

    tf = mskpy.cal2time(dates[1])
    rf, vf = mskpy.getxyz('encke', tf)
    tf_et = time2et(tf)

    dt = (tf - ti).sec
    init = State(ri * u.km, vi * (u.km / u.s), ti_et)
    final = State(rf * u.km, vf * (u.km / u.s), tf_et)
    print('init\n', init)
    print('final\n', final)
    print('dt\n', dt)

    print('\n--- beta = 0.0 ---')
    k = Kepler()
    final = k.integrate(init, dt)
    print('\nfinal\n', final)
    print('\nerror\n', final.r - rf)

#    print('\n--- beta = 0.1 ---')
#    final = k.integrate(init, dt, beta=0.1)
#    print('\nfinal\n', final)
#    print('\ndiff\n', final.r - init.r)
#    
#    print('\n--- beta = 0.0 ---')
#    r = Ra15(debug=True*False)
#    final = r.integrate(init, dt)
#    print('\nfinal\n', final)
#    print('\nerror\n', final.r - rf)
#    print('\nn_sequences ', r.n_sequences)

    for dt in (10000, 200000):
        print('\n', dt)
        tf = ti + dt * u.s
        rf, vf = mskpy.getxyz('encke', tf)
        print(State(rf * u.km, vf * (u.km / u.s), tf))
        r = Ra15(debug=False, planets=[5])
        print(r.integrate(init, dt))
        print(r.n_fcalls)
        print(Kepler().integrate(init, dt))

#    print('\n--- test ---')
#    r = Ra15(tol=1e-5, iss=100, debug=True)
#    init = State([1000, 0, 0] * u.km, [0, 10, 0] * u.km / u.s, 0)
#    final = r.integrate(init, 864000)
#    print('\nfinal\n', final)
#    import numpy as np
#    print(np.sqrt(np.sum((final.r - init.r)**2)))

