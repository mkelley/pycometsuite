"""
core
====

Classes
-------
Integrator

"""

import abc


class Integrator(abc.ABC):
    def __init__(self):
        pass

    def __repr__(self):
        return "<cometsuite.integrator: {}>".format(self.__str__())

    def _parameter_str(self):
        """Class parameters formatted as comma-separated key=value pairs.

        The string must be usable by eval().

        """
        return ""

    def __str__(self):
        cls = type(self)
        return f"{cls.__name__}({self._parameter_str()})"

    @abc.abstractmethod
    def integrate(self, init, dt, beta=0):
        """Propagate a state into the future.


        Parameters
        ----------
        init : State
            The initial state vector.

        dt : float
            The time offset. [seconds]

        beta : float, optional
            Reduce GM by the factor `(1 - beta)`.


        Returns
        -------
        final : State
          The final state vector.

        """
        pass
