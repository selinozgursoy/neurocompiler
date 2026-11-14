import pytest

import neurograph as ng


@pytest.fixture(scope="session")
def programs():
    return {name: ng.compile_graph(ng.load_preset(name)) for name in ng.PRESETS}
