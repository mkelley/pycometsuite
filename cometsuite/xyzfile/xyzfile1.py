import io
import numpy as np
import yaml
from . import XYZFileBase
from .. import __version__

__all__ = [
    'XYZFile1',

    'params2header',

    'header_template',
    'params_template',
]

header_template = ("""
cometsuite: {}
header: 1.0

date: None
comet:
  name: None
  kernel: None
  r: None
  v: None

syndynes: False
label: None
pfunc:
  age: None
  radius: None
  composition: None
  density_scale: None
  speed: None
  speed_scale: None
  vhat: None
nparticles: 0

integrator: None
box: -1

save: [ radius, graindensity, beta, age, origin, r_i, v_ej, r_f ]
...
""".format(__version__))

params_template = yaml.load(header_template)

header_example = ("""
cometsuite: 1.0.0
header: 1.0

date: 1997-07-14 01:00
comet:
  name: encke
  kernel: encke.bsp
  r: [  4.36912066e+07,  -1.64068210e+08,  -2.74004295e+07]
  v: [ 26.40437448, -21.02386471,  -1.63339555]

syndynes: False
label: None
pfunc:
  age: Uniform(x0=0, x1=31536000)
  radius: Log(x0=0, x1=3)
  composition: Geometric(rho0=1)
  density_scale: UnityScaler()
  speed: Delta(x0=0.3)
  speed_scale: SpeedRh(k=-0.5) * SpeedRadius(k=-0.5)
  vhat: Isotropic()
nparticles: 1000000

integrator: Kepler(GM=1.3274935144e+20)
box: -1

save: [ radius, graindensity, beta, age, origin, r_i, v_ej, r_f ]
units: [ micron, g/cm^3, none, s, deg, km, km/s, km ]
data: [ d(radius), d(graindensity), d(beta), d(age), "d[2](origin)", "d[3](r_i)", "d[3](v_ej)", "d[3](r_f)" ]
...
""")

class XYZFile1(XYZFileBase):
    """Open a CometSuite data file for I/O.

    inf = XYZFile1(filename, 'w', sim)
    inf = XYZFile1(filename, 'r')

    Parameters
    ----------
    filename : string
      The file name.
    mode : string
      'r' for read mode, 'w' for write mode.
    sim : Simulation
      Initialize `XYZFile1` with this simulation's parameter set.
    verbose : bool, optional
      When `False`, feedback to the user will generally be supressed.

    Attributes
    ----------

    """

    def __init__(self, filename, mode, *args, **keywords):
        from ..simulation import Simulation

        self.header = ''
        self.nread = 0
        self.nwritten = 0
        self.dtype = None
        self.start = 0
        self.params = None
        self.verbose = keywords.get('verbose', True)

        if mode == 'w':
            # test for required keywords
            sim = args[0]
            self.params = sim.params
            self._file = open(filename, 'wb')
            self._write_header()
        elif mode == 'r':
            self._file = open(filename, 'rb')
            self._read_header()
        else:
            raise ValueError("Mode must be 'r' (read) or 'w' (write).")
        self._setup_dtype()

    def _setup_dtype(self):
        """Setup numpy record array dtype."""
        import re
        dtype = []
        dataRe = re.compile('(([idc])(\[([0-9]+)\])?\((\w+)\))', re.IGNORECASE)
        tr = dict(d='<f8', i='<i4', c='S')
        data = ' '.join(save2data(self.params['save']))
        for d in dataRe.findall(data):
            dtype.append((d[4], d[3] + tr[d[1]]))
        self.dtype = np.dtype(dtype)

    def _read_header(self):
        """Read in header."""

        line = ''
        while line != '...\n':
            if self.tell() > 3000:
                raise OSError("Cannot find end of YAML header.")
            self.header += line
            line = self.readline().decode('ascii')
        self.params = yaml.load(self.header)

        # save this position as the location of the first particle
        self.start = self.tell()

    def __next__(self):
        p = np.fromfile(self._file, dtype=self.dtype, count=1)
        if len(p) == 1:
            self.nread += 1
            return p
        else:
            raise StopIteration

    def read_simulation(self, n=None):
        """Read particles from the file.

        Parameters
        ----------
        n : int or long, optional
          Read `n` particles.  If `n` is `None`, read as many
          particles as are expected, based on the data header.

        Returns
        -------
        p : numpy record array
          The array of particle data.

        """

        from ..simulation import Simulation

        if n is None:
            n = self.params['nparticles'] - self.nread
        if self.verbose:
            print('[xyzfile] Reading {} particles.'.format(n))
        sim = Simulation(verbose=self.verbose)
        sim.params = self.params
        sim.particles = np.fromfile(self._file, dtype=self.dtype,
                                    count=n).view(np.recarray)
        self.nread += len(sim.particles)
        if self.verbose:
            print('[xyzfile] {} particles read.'.format(len(sim.particles)))
        return sim

    def _write_header(self):
        from datetime import datetime
        now = datetime.isoformat(datetime.utcnow())
        self.write(("# {}\n".format(now)).encode('ascii'))
        self.write(params2header(self.params).encode('ascii'))

    def write_particles(self, particles):
        """Write a particle or list of particles to the file.

        Parameters
        ----------
        particles : dictionary-like
          A dictionary-like (e.g., dict or numpy record array) set of
          parameters for each particle.

        Examples
        --------
        p = dict(radius=np.random.rand(10), age=np.random.rand(10) * 100)
        sim = Simulation()
        sim.params.update(comet='encke', kernel='encke.bsp', date=2450000.5,
                          nparticles=len(p['radius']), save=['radius', 'age'])
        outf = cs.XYZFile1('test.xyz', 'w', sim)
        outf.write_particles(p)
        outf.close()

        """

        # determine the number to write and setup record array
        size = (np.size(particles[self.dtype.names[0]]),)
        p = np.recarray(size, dtype=self.dtype)
        for k in self.dtype.names:
            if k == 'label':
                # null pad the labels
                s = self.dtype['label'].itemsize
                nullpad = lambda x: '{}{}'.format(x, '\0' * s)
                p['label'] = list(map(nullpad, particles['label']))
            else:
                p[k] = particles[k]
        p.tofile(self._file)
        self.nwritten += len(p)

def save2units(save):
    """Convert a 'save' list into units."""
    units = dict(radius='micron', graindensity='g/cm^3', beta='none',
                 age='s', origin='deg', v_ej='km/s', r_i='km',
                 v_i='km/s', t_i='s', r_f='km', v_f='km/s', t_f='s',
                 label='none')
    return [units[k] for k in save]

def save2data(save):
    """Convert a 'save' list into a data description."""
    data = dict(radius='d(radius)', graindensity='d(graindensity)',
                beta='d(beta)', age='d(age)',
                origin='d[2](origin)', v_ej='d[3](v_ej)',
                r_i='d[3](r_i)', v_i='d[3](v_i)', t_i='d(t_i)',
                r_f='d[3](r_f)', v_f='d[3](v_f)', t_f='d(t_f)',
                label='c[16](label)')
    return [data[k] for k in save]

def params2header(params):
    """A Simulation parameter set as an XYZFile1 header string.

    Parameters
    ----------
    params : dict
      The parameter set.

    Returns
    -------
    header : string
      The XYZFile1 header, in YAML format.

    """

    header = yaml.dump(params, default_flow_style=False)
    header += yaml.dump(dict(units=save2units(params['save']),
                             data=save2data(params['save'])),
                        default_flow_style=False)
    #header += "units: {}\ndata: {}\n...\n".format(
    #    save2units(params['save']), save2data(params['save']))
    header += '...\n'

    return header

