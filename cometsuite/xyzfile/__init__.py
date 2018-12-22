"""
xyzfile --- Cometsuite save files
==================================

Proprietary code: do not share without express permission.

Use XYZFile or XYZFile1 for v1 files.  Use XYZFile0 for v0.

.. autosummary::
   :toctree: generated/

   Classes
   -------
   XYZFileBase
   XYZFile
   XYZFile0
   XYZFile1
   
   Functions
   ---------
   params2header
   xyz_version

   Variables
   ---------
   header_template
   params_template

"""

__all__ = [
    'XYZFile'
]

class XYZFileBase(object):
    """Abstract base class for simulation files."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.__exit__(*args)
    
    def close(self):
        self._file.close()
        
    def read(self, b):
        return self._file.read(b)

    def readline(self):
        return self._file.readline()

    def readlines(self, b):
        return self._file.readlines(b)
        
    def seek(self):
        return self._file.tell()
    
    def tell(self):
        return self._file.tell()
    
    def write(self, b):
        return self._file.write(b)
        
    def writelines(self, b):
        return self._file.writelines(b)

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

    with open(filename, 'rb') as inf:
        import re

        # Test for v0:
        with open(filename, 'rb') as inf:
            # should start with a version comment
            line = inf.readline()

        match = re.findall(b'^#\s+CometSuite\s+(\d+)(\.\d+)(\.\d+)?\s*\n',
                           line, re.IGNORECASE)
        if len(match) == 1:
            return ''.join(match[0])

        # try reading with XYZFile v1
        try:
            inf = XYZFile1(filename, 'r')
        except IOError:
            inf.close()
            print ("No v0 version comment, and cannot open with XYZFile1.\n"
                   "Assuming v0")
            return '0'

        version = inf.params['cometsuite']

        return version


# update module docstring
from mskpy.util import autodoc
autodoc(globals())
del autodoc
