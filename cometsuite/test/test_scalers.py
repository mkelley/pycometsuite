import pytest
from .. import scalers as sc
from . import sim_radius_uniform


def test_constant_factor(sim_radius_uniform):
    s = sc.ConstantFactor(10)
    p = sim_radius_uniform.particles
    assert all(s.scale(p) == 10)
    assert str(s) == "ConstantFactor(10)"


def test_unity_scaler(sim_radius_uniform):
    s = sc.UnityScaler()
    p = sim_radius_uniform.particles
    assert all(s.scale(p) == 1)


def test_scaler_mult(sim_radius_uniform):
    s = sc.ConstantFactor(10) * sc.ConstantFactor(2)
    p = sim_radius_uniform.particles
    assert all(s.scale(p) == 20)


class TestCompositeScaler:
    def test_init(self, sim_radius_uniform):
        s = sc.CompositeScaler(sc.ConstantFactor(10), sc.ConstantFactor(2))
        p = sim_radius_uniform.particles
        assert all(s.scale(p) == 20)

    def test_mult(self, sim_radius_uniform):
        s = sc.CompositeScaler(sc.ConstantFactor(10)) * sc.ConstantFactor(2)
        p = sim_radius_uniform.particles
        assert all(s.scale(p) == 20)
