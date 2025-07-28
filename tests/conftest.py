"""
Global fixtures for the tests.
"""

import pytest


@pytest.fixture(scope="session")
def global_fixture():
    yield "global_fixture"
