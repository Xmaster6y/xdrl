import pytest

from xdrl.core import explain


@pytest.mark.unit
def test_explain():
    assert explain(1, 1) == 0
