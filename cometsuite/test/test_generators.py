import numpy as np
from numpy import pi
import pytest
import cometsuite.generators as g
from cometsuite.generators import *

def test_CosineAngle():
    # make the test deterministic
    # struct.unpack("<L", np.random.bytes(4))[0]
    np.random.seed(1448982574)
    
    N = 1000
    u = CosineAngle(0, pi / 2)
    th = np.array([next(u) for i in range(N)])
    th.sort()

    assert th[0] >= 0
    assert th[-1] <= pi / 2

    # compare CDF to ideal case
    f = np.cumsum(np.ones_like(th)) / N
    d = f - (1 - np.cos(th)**2)
    assert np.mean(d) < 2 * np.std(d)

def test_CosineAngle_limits():
    # make the test deterministic
    # struct.unpack("<L", np.random.bytes(4))[0]
    np.random.seed(1448982574)
    
    N = 1000
    u = CosineAngle(pi / 6, pi / 4)
    th = np.array([next(u) for i in range(N)])
    th.sort()

    assert th[0] >= pi / 6
    assert th[-1] <= pi / 4

    # compare CDF to ideal case
    f = np.cumsum(np.ones_like(th)) / N
    d = f - (1 - np.cos(th)**2)
    assert np.mean(d) < 2 * np.std(d)

def test_Delta():
    N = 1000
    u = Delta(5)
    x = np.array([next(u) for i in range(N)])
    assert np.all(x == 5)

def test_Grid():
    N = 1000
    u = Grid(0, 5, N)
    x = np.array([next(u) for i in range(N)])
    assert np.all(x == np.linspace(0, 5, N))

    with pytest.raises(StopIteration):
        next(u)

def test_Grid_endpoint():
    N = 1000
    u = Grid(0, 5, N, endpoint=False)
    x = np.array([next(u) for i in range(N)])
    assert np.all(x == np.linspace(0, 5, N, endpoint=False))

    with pytest.raises(StopIteration):
        next(u)

def test_Grid_log():
    N = 1000
    u = Grid(-1, 5, N, log=True)
    x = np.array([next(u) for i in range(N)])
    assert np.all(x == np.logspace(-1, 5, N))

    with pytest.raises(StopIteration):
        next(u)

def test_Grid_cycle():
    N = 100
    u = Grid(0, 5, N, cycle=2)
    x = np.array([next(u) for i in range(2 * N)])
    assert np.all(x == np.tile(np.linspace(0, 5, N), 2))

    with pytest.raises(StopIteration):
        next(u)

def test_Grid_repeat():
    N = 100
    u = Grid(0, 5, N, repeat=2)
    x = np.array([next(u) for i in range(2 * N)])
    assert np.all(x == np.repeat(np.linspace(0, 5, N), 2))

    with pytest.raises(StopIteration):
        next(u)

def test_Log():
    # make the test deterministic
    # struct.unpack("<L", np.random.bytes(4))[0]
    np.random.seed(1448982574)

    N = 1000
    u = Log(-1, 1)
    x = np.array([next(u) for i in range(N)])
    x.sort()

    assert x[0] >= 0.1
    assert x[-1] <= 10

    # compare CDF to ideal case
    f = np.cumsum(np.ones_like(x)) / N
    d = f - np.log(10) * np.log(x)
    assert np.mean(d) < 2 * np.std(d)

def test_Normal():
    from scipy.special import erfc
    # make the test deterministic
    # struct.unpack("<L", np.random.bytes(4))[0]
    np.random.seed(1448982574)

    N = 1000
    u = Normal()
    x = np.array([next(u) for i in range(N)])
    x.sort()
    
    # compare CDF to ideal case
    f = np.cumsum(np.ones_like(x)) / N
    d = f - (2 - erfc(x)) / 2
    assert np.mean(d) < 2 * np.std(d)

def test_Normal_limits():
    from scipy.special import erfc
    # make the test deterministic
    # struct.unpack("<L", np.random.bytes(4))[0]
    np.random.seed(1448982574)

    N = 1000
    u = Normal(x0=0)
    x = np.array([next(u) for i in range(N)])
    x.sort()
    
    # compare CDF to ideal case
    f = np.cumsum(np.ones_like(x)) / N
    d = f - (2 - erfc(x)) / 2
    assert np.mean(d) < 2 * np.std(d)

def test_Sequence():
    N = 100
    s = np.random.rand(N)
    u = Sequence(s)
    x = np.array([next(u) for i in range(N)])
    assert np.all(s == x)

    with pytest.raises(StopIteration):
        next(u)

def test_Sequence_cycle():
    N = 100
    s = np.random.rand(N)
    u = Sequence(s, cycle=2)
    x = np.array([next(u) for i in range(2 * N)])
    assert np.all(x == np.tile(s, 2))

    with pytest.raises(StopIteration):
        next(u)

def test_Sequence_repeat():
    N = 100
    s = np.random.rand(N)
    u = Sequence(s, repeat=2)
    x = np.array([next(u) for i in range(2 * N)])
    assert np.all(x == np.repeat(s, 2))

    with pytest.raises(StopIteration):
        next(u)
        
def test_Uniform():
    # make the test deterministic
    # struct.unpack("<L", np.random.bytes(4))[0]
    np.random.seed(1448982574)

    N = 1000
    u = Uniform(0, 5)
    x = np.array([next(u) for i in range(N)])
    x.sort()

    assert x[0] >= 0
    assert x[-1] <= 5

    # compare CDF to ideal case
    f = np.cumsum(np.ones_like(x)) / N
    d = f - x
    assert np.mean(d) < 2 * np.std(d)

def test_UniformAngle():
    # make the test deterministic
    # struct.unpack("<L", np.random.bytes(4))[0]
    np.random.seed(1448982574)
    
    N = 1000
    u = UniformAngle(0, pi)
    th = np.array([next(u) for i in range(N)])
    th.sort()

    assert th[0] >= 0
    assert th[-1] <= pi

    # compare CDF to ideal case
    f = np.cumsum(np.ones_like(th)) / N
    d = f - (1 - np.cos(th)) / 2
    assert np.mean(d) < 2 * np.std(d)

def test_UniformAngle_limits():
    # make the test deterministic
    # struct.unpack("<L", np.random.bytes(4))[0]
    np.random.seed(1448982574)
    
    N = 1000
    u = UniformAngle(0.2, 2)
    th = np.array([next(u) for i in range(N)])
    th.sort()

    assert th[0] >= 0
    assert th[-1] <= pi

    # compare CDF to ideal case
    f = np.cumsum(np.ones_like(th)) / N
    d = f - (1 - np.cos(th)) / 2
    assert np.mean(d) < 2 * np.std(d)

def test_Isotropic():
    from cometsuite.state import State
    np.random.seed(1448982574)

    N = 1000
    u = Isotropic()
    init = State([1.0, 0, 0], [0, 1.0, 0], 2458056.5)
    v, origin = u.next(init, N)

    # expectation value for the vector sum magnitude is sqrt(N)
    assert np.sqrt(np.sum(v.sum(0)**2)) < 2 * np.sqrt(N)

def test_UniformLatitude():
    from cometsuite.state import State
    np.random.seed(1448982574)

    N = 1000
    init = State([1.0, 0, 0], [0, 1.0, 0], 2458056.5)
    pole = np.array([0, 0, 1.0])
    u = UniformLatitude(np.radians((-5, 5)), pole=pole)
    v, origin = u.next(init, N)

    # expectation value for the vector sum magnitude is
    # |1/3 1/3 sin(5 deg)| * sqrt(N) ?
    d = np.sqrt(np.sum(np.array([1/3, 1/3, np.sin(np.radians(5))])**2))
    assert np.sqrt(np.sum(v.sum(0)**2)) < 2 * d * np.sqrt(N)

    assert min(origin[:, 1]) >= -5
    assert max(origin[:, 1]) <= 5

def test_Sunward():
    from mskpy.util import mhat
    from cometsuite.state import State
    
    np.random.seed(1448982574)

    N = 1000

    # generate N random Sun vectors
    init = list([State([x, y, z], [0, 1.0, 0], 2458056.5)
                 for x, y, z in np.random.rand(N, 3)])
    s = -mhat(list((i.r for i in init)))[1]
    
    # all vectors should align with the sunward direction
    u = Sunward(w=0)
    v, origin = u.next(init, N)
    assert np.allclose(v, s)

    # all vectors should be within 25 degrees from sunward direction
    u = Sunward(w=np.radians(50))
    v, origin = u.next(init, N)
    assert not np.allclose(v, s)

    cth = np.sum(s * v, -1)
    assert min(cth) >= np.cos(np.radians(25))
    
