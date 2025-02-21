"""
rundynamics - Generate and integrate particle states.
=====================================================

Functions
---------

run

"""

import time
from typing import Union
import numpy as np
from .xyzfile import XYZFile
from .simulation import Simulation


def run(
    pgen, integrator, xyzfile=None, limit=None, seed=None
) -> Union[Simulation, int]:
    """Generate and integrate particle states.

    `run` will respect `pgen.nparticles`, therefore it must be correctly set.

    If the simulation is 'make comet', the first particle will have the position
    of the comet.  The other parameters will all be zero.


    Parameters
    ----------
    pgen : ParticleGenerator
        The particle generator.

    integrator : Integrator
        The particle state integrator.

    xyzfile : XYZFile or string, optional
        Save particle states in this `XYZFile`.

    limit : int, optional
        Limit the number of states to integrate to `n`.

    seed : int or array, optional
        Seed for `np.random.seed`.


    Returns
    -------
    n : int, optional
        When `xyzfile` is set, the number of particle states integrated is
        returned.

    sim : Simulation, optional
        When `xyzfile` is `None`, the results are returned as a `Simulation`.

    """

    np.random.seed(seed)

    if limit is None:
        limit = float("inf")
    N = pgen.nparticles if pgen.nparticles < limit else limit
    print("[run] Expecting {} particles".format(N))

    sim = pgen.sim()
    sim.params["integrator"] = str(integrator)

    if xyzfile is None:
        outf = None
        sim.init_particles()
    else:
        if isinstance(xyzfile, XYZFile):
            outf = xyzfile
        else:
            print("[run] Opening {} for output.".format(xyzfile))
            outf = XYZFile(xyzfile, "w", sim)

    def add_particle(p, xyzfile, sim, outf):
        if xyzfile is None:
            sim[i] = p
        else:
            outf.write_particles(p)

    start_time = time.time()
    last_time = start_time
    last_status = 0
    status_step = 1000
    i = 0
    for i, p in enumerate(pgen):
        p.final = integrator.integrate(p.init, -p.t_i, beta=p.beta)
        add_particle(p, xyzfile, sim, outf)

        if i >= limit:
            print("[run] limit reached.")
            break

        if ((i + 1) % status_step) == 0:
            now = time.time()
            dt = now - last_time
            rate = dt / float(status_step - last_status)
            t_est = now + (pgen.nparticles - i) * rate
            t_est = time.strftime("%d %b %H:%M", time.localtime(t_est))
            print(
                (
                    "[run] {} integrated, {:.3g} s/particle,"
                    " complete at {}".format(i + 1, rate, t_est)
                )
            )

            last_time = now
            last_status = i + 1
            status_step *= 2

    i += 1
    print("[run] {} particle states integrated".format(i))
    dt = time.time() - start_time
    print("[run] Overall, {} seconds per particle".format(dt / float(i)))

    if xyzfile is None:
        return sim
    else:
        outf.close()
        return i
