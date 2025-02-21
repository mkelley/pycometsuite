__all__ = ["ParameterWeight"]

from .core import Scaler


class ParameterWeight(Scaler):
    """Scale value based on a parameter.


    Parameters
    ----------
    key : string
        The particle parameter key that defines the scale factor, e.g., 'age'.


    Methods
    -------
    scale : Scale factor.

    """

    def __init__(self, key):
        self.key = key

    def __str__(self):
        return 'ParameterWeight("{}")'.format(self.key)

    def formula(self):
        return "W = {}".format(self.key)

    def scale(self, p):
        return p[self.key]
