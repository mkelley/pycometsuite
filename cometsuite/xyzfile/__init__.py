__all__ = ["XYZFile", "XYZFile1", "XYZFile0", "xyz_version"]

from .xyzfile0 import *
from .xyzfile1 import *
from .xyzfile1 import XYZFile1 as XYZFile


def xyz_version(filename):
    """Determine the CometSuite version of an xyzfile.

    Parameters
    ----------
    filename : string

    Returns
    -------
    version : string
      '0' for pre 1.0, '1' otherwise.

    """

    with open(filename, "rb") as inf:
        import re

        # Test for v0:
        with open(filename, "rb") as inf:
            # should start with a version comment
            line = inf.readline()

        match = re.findall(
            r"^#\s+CometSuite\s+(\d+)(\.\d+)(\.\d+)?\s*\n", line, re.IGNORECASE
        )
        if len(match) == 1:
            return "".join(match[0])

        # try reading with XYZFile v1
        try:
            inf = XYZFile1(filename, "r")
        except IOError:
            inf.close()
            print(
                "No v0 version comment, and cannot open with XYZFile1.\n" "Assuming v0"
            )
            return "0"

        version = inf.params["cometsuite"]

        return version
