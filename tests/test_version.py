import importlib
import sys
import types

import welt_io_openai_agents


def test_generated_version_is_exposed() -> None:
    assert welt_io_openai_agents.__version__ != "0.0.0+unknown"


def test_missing_version_module_falls_back_to_unknown() -> None:
    # Simulates a source tree without the hatch-vcs-generated _version.py:
    # a _version module carrying no __version__ raises the same ImportError
    # on `from ._version import __version__` as the missing file does.
    version_module = sys.modules["welt_io_openai_agents._version"]
    sys.modules["welt_io_openai_agents._version"] = types.ModuleType(
        "welt_io_openai_agents._version"
    )
    try:
        assert importlib.reload(welt_io_openai_agents).__version__ == "0.0.0+unknown"
    finally:
        sys.modules["welt_io_openai_agents._version"] = version_module
        importlib.reload(welt_io_openai_agents)
