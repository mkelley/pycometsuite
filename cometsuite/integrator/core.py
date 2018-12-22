"""
core
====

Classes
-------
Integrator

"""

class Integrator(object):
    def __init__(self):
        pass

    def __repr__(self):
        return '<cometsuite.integrator: {}>'.format(self.__str__())

    def __str__(self):
        return 'Integrator()'

    def integrate(self, state, dt, beta=0):
        """Propagate a state into the future.

        Parameters
        ----------
        init : State
          The state vector.
        dt : float
          The time offset. [seconds]
        beta : float, optional
          Reduce GM by the factor `(1 - beta)`.

        Returns
        -------
        final : State
          The final state.

        """
        pass
