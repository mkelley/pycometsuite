"""
xyzfile0
----------
"""

import io
import numpy as np
from . import XYZFileBase

__all__ = ['XYZFile0']

class XYZFile0(XYZFileBase):
    """Reads xyzfile format v0.

    Simluation has been upgraded beyond v0.
    `XYZFile0.read_simluation` works, but the returned sim will only
    be partially useful: the parameters will not make sense, but the
    particle array will be OK.

    Parameters
    ----------
    filename : string
      The name of the file.
    sim : Simulation
      Initialize `XYZFile0` with this `Simulation`.
    mode : string
      Access mode: 'r' or 'w'.
    verbose : bool, optional
      When `False`, feedback to the user will generally be supressed.

    Attributes
    ----------

    """

    _writeRequiredKeywords = ['comet', 'kernel', 'jd', 'nparticles',
                              'datalist']
    _csFormats = dict(radius='d(radius)', graindensity='d(graindensity)',
                      beta='d(beta)', age='d(age)', origin='d[2](origin)',
                      v_ej='d[3](v_ej)', r_i='d[3](r_i)', v_i='d[3](v_i)',
                      t_i='d(t_i)', r_f='d[3](r_f)', v_f='d[3](v_f)',
                      t_f='d(t_f)', label='c[16](label)')

    # we will need to keep track of parameters, but won't keep track
    # of particles
    params = None

    def __init__(self, filename, mode, *args, **keywords):
        from ..simulation import Simulation

        self.header = ''
        self.nread = 0
        self.nwritten = 0
        self.datalist = ()
        self.rec = None
        self.start = 0
        self.params = dict()
        self.verbose = keywords.get('verbose', True)

        if mode == 'w':
            # test for required keywords
            sim = args[0]
            for k in self._writeRequiredKeywords:
                if k not in vars(sim).keys():
                    raise ValueError(
                        "sim is missing one or more required keywords: " +
                        ', '.join(self._writeRequiredKeywords))
            for k in sim.parameters:
                #setattr(self.params, k, sim[k])
                self.params[k] = sim[k]
            self._file = open(filename, 'wb')
            self._writeHeader()
        elif mode == 'r':
            self._file = open(filename, 'rb')
            self._readHeader()
        else:
            raise ValueError("Mode must be 'r' (read) or 'w' (write).")

    def _data2rec(self):
        """Convert DATA line to a numpy record array."""
        import re
        dtype = []
        dataRe = re.compile('(([idc])(\[([0-9]+)\])?\((\w+)\))', re.IGNORECASE)
        tr = dict(d='<f8', i='<i4', c='S')
        for d in dataRe.findall(self.params['data']):
            dtype.append((d[4], d[3] + tr[d[1]]))
        return np.dtype(dtype)

    def _readHeader(self):
        """Read in header and prepare XYZFile0 for particle reading."""
        import re

        # the first line is the program name and version
        line = self.readline().encode('ascii')
        self.header += line
        if (('cometsuite' not in line.lower()) and
            ('rundynamics' not in line.lower())):
            self.close()
            raise OSError("Does not appear to be a rundynamics file.")

        # the second line is the time stamp
        self.header += self.readline().encode('ascii')

        # the parameter format
        paramRe = re.compile('^(\w+):\s*(.*)', re.IGNORECASE)

        # the comment format
        commentRe = re.compile('^#')

        # for parsing data lists
        dataRe = re.compile('([idc](\[[0-9]\])?\(\w+\))', re.IGNORECASE)

        # loop through all the parameters, stopping at the beginning of
        # the data description
        while '# data file description' not in line.lower():
            line = self.readline().encode('ascii')
            self.header += line
            if commentRe.match(line) == None:
                # the line is not a comment
                match = paramRe.match(line)
                if match is not None:
                    # the line has a parameter and value
                    parameter, value = match.groups()
                    parameter = parameter.upper().strip()
                    if parameter == 'PROGRAM':
                        if value.lower().find('syndynes') == 0:
                            self.params['program'] = 'syndynes'
                        elif value.lower().find('make comet') == 0:
                            self.params['program'] = 'make comet'
                        elif value.lower().find('integrate xyz') == 0:
                            self.params['program'] = 'integrate xyz'
                        else:
                            self.close()
                            raise OSError("I don't understand program " +
                                          "parameter:" + value)
                    elif parameter == 'COMET':
                        self.params['comet'] = value
                    elif parameter == 'KERNEL':
                        self.params['kernel'] = value
                    elif parameter == 'JD':
                        self.params['jd'] = np.double(value)
                    elif parameter == 'XYZFILE':
                        self.params['xyzfile'] = value
                    elif parameter == 'LABEL':
                        self.params['labelprefix'] = value
                    elif parameter == 'PFUNC':
                        self.params['pfunc'] = value
                    elif parameter == 'TOL':
                        self.params['tol'] = float(value)
                    elif parameter == 'PLANETS':
                        self.params['planets'] = int(value)
                    elif parameter == 'PLANETLOOKUP':
                        self.params['planetlookup'] = self._yorn(value)
                    elif parameter == 'CLOSEAPPROACHES':
                        self.params['closeapproaches'] = self._yorn(value)
                    elif parameter == 'BOX':
                        self.params['box'] = float(value)
                    elif parameter == 'LTT':
                        self.params['ltt'] = self._yorn(value)
                    elif parameter == 'SAVE':
                        self.params['save'] = value.split()
                    elif parameter == 'BETA':
                        self.params['synbeta'] = np.array(
                            re.split('[\s,]+', value), dtype=np.float)
                    elif parameter == 'NDAYS':
                        self.params['ndays'] = int(value)
                    elif parameter == 'STEPS':
                        self.params['steps'] = int(value)
                    elif parameter == 'ORBIT':
                        self.params['orbit'] = float(value)
                    elif parameter == 'NPARTICLES':
                        self.params['nparticles'] = int(value)
    
        # if there is an orbit, add one more beta for the orbit
        if self.params['orbit'] > 0:
            self.params['synbeta'] = np.concatenate((self.params['synbeta'], (-99, )))

        # nparticles for a syndyne simulation
        self.params['nbeta'] = self.params['synbeta'].size
        if self.params['program'] == 'syndynes':
            self.params['nparticles'] = self.params['nbeta'] * self.params['steps']

        # read in UNITS and DATA
        line = self.readline().encode('ascii')
        self.header += line
        self.params['units'] = line.partition(':')[2].strip()

        line = self.readline().encode('ascii')
        self.header += line
        self.params['data'] = line.partition(':')[2].strip()

        # save this position as the location of the first particle
        self.start = self.tell()

        # setup data record array for reading
        self.rec = self._data2rec()
        self.params['datalist'] = self.rec.names

    def _yorn(self, s):
        """Yes or no.

        Parameters
        ----------
        s : string, int, bool
          Input to inspect.

        Returns
        -------
        answer : bool
          `True` if string s is 'y', 'yes', 't', 'true', or a non-zero
          number.  Otherwise, returns `False`.  Whitespace characters
          are stripped from `s`.

        """

        a = s.lower().strip()

        if a in ['y', 'yes', 't', 'true']:
            return True

        try:
            return bool(int(a))
        except ValueError:
            pass

        return False

    def __next__(self):
        p = np.fromfile(self._file, dtype=self.rec, count=1)
        if len(v) == 1:
            self.nread += 1
            p['age'] /= 86400
            return p
        else:
            raise StopIteration

    def read_simulation(self, n=None):
        """Read particles from the file.

        Default is to read all.

        Usage::
          p = readParticles([n=])

        Parameters
        ----------

        n : int or long, optional
          Read ``n`` particles.  If n is None, read as many particles
          as are expected, based on the data header.

        Returns
        -------

        p : numpy record array
          The array of particle data.

        """

        from ..simulation import Simulation

        if n is None:
            n = self.params['nparticles'] - self.nread

        sim = Simulation(**self.params)
        sim.particles = np.fromfile(self._file, dtype=self.rec,
                                    count=n).view(np.recarray)
        sim.particles['age'] /= 86400
        self.nread += len(sim.particles)

        return sim

    def _writeHeader(self):
        from datetime import datetime
        now = datetime.isoformat(datetime.datetime.utcnow())
        self.header = "# cometsuite : cometsuite.XYZFile0\n# {}\n".format(now)
        self.header += self.params['xyz_header']
        self.header += ("""# data file description
UNITS: {}
DATA: {}
""".format(self.params['units'], self.params['data']))

        self.write(self.header.encode('ascii'))

        # setup data record array for writing
        self.rec = self._data2rec()

    def writeParticles(self, particles):
        """Write a particle or list of particles to the file.

        Parameters
        ----------
        particles : dictionary-like
          A dictionary-like (e.g., dict or numpy record array) set of
          parameters for each particle.

        Examples
        --------
        p = dict(radius=np.random.rand(10), age=np.random.rand(10) * 100)
        outf = cs.XYZFile0('test.xyz', 'w', comet='encke',
                           kernel='encke.bsp', jd=2450000.5,
                           nparticles=len(p['radius']),
                           datalist=('radius', 'age'))
        outf.writeParticles(p)
        outf.close()

        """
        # determine the number to write and setup record array
        size = (np.size(particles[self.rec.names[0]]),)
        p = np.recarray(size, dtype=self.rec)
        for k in self.rec.names:
            if k == 'label':
                # null pad the labels
                s = self.rec['label'].itemsize
                nullpad = lambda x: '{}{}'.format(x, '\0' * s)
                p['label'] = list(map(nullpad, particles['label']))
            else:
                p[k] = particles[k]
        p.tofile(self._file)
        self.nwritten += len(p)
