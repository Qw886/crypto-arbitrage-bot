"""Keep network-free tests independent of whichever proxy runs the workstation."""

import os
import runpy
import urllib.request
from pathlib import Path

import pytest

_OFFLINE_RUNTIME = Path(__file__).with_name("offline_runtime").resolve()
# Install during collection; test-local socket mocks cannot bypass audit events.
runpy.run_path(str(_OFFLINE_RUNTIME / "sitecustomize.py"))


@pytest.fixture(scope="session", autouse=True)
def inherit_configured_pythonpath(pytestconfig):
    """Native test children import the same configured source as their parent.

    pytest's pythonpath option changes sys.path only. Absolute environment paths
    also work after a test changes cwd, and honor isolated -o pythonpath runs.
    This applies only to the test session and restores the caller's environment.
    """
    original = os.environ.get("PYTHONPATH")
    configured = [str(_OFFLINE_RUNTIME)]
    configured.extend(str(Path(path).resolve()) for path in pytestconfig.getini("pythonpath"))
    inherited = [path for path in (original or "").split(os.pathsep) if path]
    os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys((*configured, *inherited)))
    try:
        yield
    finally:
        if original is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original


@pytest.fixture(autouse=True)
def isolate_proxy_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name, raising=False)
    # Real Windows settings are now supported in production. Offline tests
    # must mock them just as they mock environment variables and TCP sockets.
    for name, fallback in (("getproxies_registry", {}), ("getproxies_macosx_sysconf", {}),
                           ("proxy_bypass_registry", False), ("proxy_bypass_macosx_sysconf", False)):
        if hasattr(urllib.request, name):
            monkeypatch.setattr(urllib.request, name, lambda *args, result=fallback: result)
