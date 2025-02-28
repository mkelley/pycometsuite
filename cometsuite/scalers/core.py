__all__ = [
    "Scaler",
    "CompositeScaler",
    "ConstantFactor",
    "UnityScaler",
]

import abc
from copy import copy
from collections import UserList

import numpy as np


class InvalidScaler(Exception):
    pass


class Scaler(abc.ABC):
    """Abstract base class for particle scale factors.

    Notes
    -----
    Particle scale factors are multiplicative.

    Radius must be in μm.

    """

    def __init__(self):
        pass

    def __mul__(self, scale):
        return CompositeScaler(self, scale)

    def __repr__(self):
        return "<" + str(self) + ">"

    @abc.abstractmethod
    def __str__(self):
        return "Scaler()"

    def copy(self):
        return copy(self)

    def formula(self):
        return ""

    @abc.abstractmethod
    def scale(self, p):
        """Scale factors for this simulation or particle."""
        return 1.0


class CompositeScaler(Scaler, UserList):
    """A collection of multiple scale factors.

    To create a `CompositeScaler`::

        total_scale = CompositeScaler(SpeedRh(), SpeedRadius())
        total_scale = SpeedRh() * SpeedRadius()

    To remove the `SpeedRh` scale::

        del total_scale[0]

    A length-one `CompositeScaler` may also be created::

        s = CompositeScaler(SpeedRh())
        s = SpeedRh() * UnityScaler()

    Iterate over the scaler functions::

        for scaler in scalers:
            scaler


    Raises
    ------
    InvalidScaler
        If a scaler is not an instance of `Scaler`, `CompositeScaler`, `float`,
        or `int`.

    """

    def __init__(self, *scales):
        _scales = []
        for sc in scales:
            if isinstance(sc, UnityScaler):
                pass
            elif isinstance(sc, CompositeScaler):
                _scales.extend([s.copy() for s in sc])
            elif isinstance(sc, Scaler):
                # must test after CompositeScaler
                _scales.append(sc.copy())
            elif isinstance(sc, (float, int)):
                _scales.append(ConstantFactor(sc))
            else:
                raise InvalidScaler(sc)

        super(Scaler, self).__init__(_scales)

    def __mul__(self, scale):
        result = self.copy()
        result *= scale
        return result

    def __imul__(self, scale):
        if isinstance(scale, Scaler):
            self.append(scale)
        elif isinstance(scale, (CompositeScaler, list, tuple)):
            self.extend(scale)
        elif isinstance(scale, (float, int)):
            self.append(ConstantFactor(scale))
        else:
            raise InvalidScaler(scale)

        return self

    def __repr__(self):
        return "CompositeScaler([{}])".format(", ".join([repr(s) for s in self]))

    def __str__(self):
        if len(self) == 0:
            return str(UnityScaler())
        else:
            return " * ".join([str(s) for s in self])

    def formula(self):
        return [s.formula() for s in self]

    def filter(self, cls, inverse=False):
        """Return scalers derived from the given classes.


        Parameters
        ----------
        cls : class or list of classes
            The class(es) to return.

        inverse : bool, optional
            Set to ``False`` to return all scalers except those of the given
            class(es).


        Returns
        -------
        scaler : CompositeScaler

        """

        def test(s):
            t = isinstance(s, cls)
            if inverse:
                return not t
            else:
                return t

        return CompositeScaler(*(s for s in self if test(s)))

    def scale(self, p):
        c = 1.0
        for s in self:
            c = c * s.scale(p)
        return c

    def scale_a(self, a):
        c = 1.0
        for s in self:
            if hasattr(s, "scale_a"):
                c = c * s.scale_a(a)
        return c

    def scale_rh(self, rh):
        c = 1.0
        for s in self:
            if hasattr(s, "scale_rh"):
                c = c * s.scale_rh(rh)
        return c


class UnityScaler(Scaler):
    """Scale factor of 1.0."""

    def __init__(self):
        pass

    def __str__(self):
        return "UnityScaler()"

    def scale(self, p):
        return np.ones_like(p.radius)

    def scale_a(self, a):
        return np.ones_like(a)

    def scale_rh(self, rh):
        return np.ones_like(rh)


class ConstantFactor(Scaler):
    """Constant scale factor."""

    def __init__(self, c):
        self.c = c

    def __str__(self):
        return "{}({})".format(type(self).__name__, self.c)

    def formula(self):
        return r"$C = {:.3g}$".format(self.c)

    def scale(self, p):
        return self.c * np.ones(len(p))
