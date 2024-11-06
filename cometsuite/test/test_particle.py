import pytest
import numpy as np
from .. import particle


class TestAmorphousCarbon:

    @pytest.mark.parametrize(
        "p, a, expected",
        [
            (0, 0.3370064329271928449, 1.904467539223512418),
            (0.06, 8.432734457879398, 1.0408506896316851),
            (0.615, 30.19252937358978173, 0.985840030802912),
        ],
    )
    def test_Qpr(self, p, a, expected):
        """Testing to make sure interp2d replacement is being used correctly."""
        ac = particle.AmorphousCarbon()
        q = ac.Qpr(a, p)
        assert np.isclose(q, expected)
