import torch
from LION.operators.Operator import Operator
from spyrit.core.torch import fwht, ifwht
from tests.helper import dotproduct_adjointness_test


def test_wht_adjointness():
    """Test WHT operator adjoint property."""
    n = 256
    x = torch.rand(n)
    y = torch.rand(n)

    class WhtOp(Operator):
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return fwht(x, dim=0)

        def adjoint(self, y: torch.Tensor) -> torch.Tensor:
            return fwht(y, dim=0)

    operator = WhtOp()
    dotproduct_adjointness_test(operator, x, y)

    torch.testing.assert_close(ifwht(operator(y)), y)
