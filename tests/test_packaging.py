"""What `pip install behaviorgpt` gives a customer: the built wheel, in a clean
venv with only the declared runtime dependencies, must import."""

import shutil
import subprocess
import sys

import pytest

from .conftest import ROOT

pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")


@pytest.fixture(scope="module")
def clean_install(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pkg")
    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp / "dist"), str(ROOT)],
        check=True,
        capture_output=True,
    )
    wheel = next((tmp / "dist").glob("*.whl"))
    venv = tmp / "venv"
    subprocess.run(
        [
            "uv",
            "venv",
            "-q",
            "-p",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            str(venv),
        ],
        check=True,
        capture_output=True,
    )
    python = venv / "bin" / "python"
    subprocess.run(
        ["uv", "pip", "install", "-q", "-p", str(python), str(wheel)],
        check=True,
        capture_output=True,
    )
    return python


def test_wheel_declares_only_runtime_deps(clean_install):
    freeze = subprocess.run(
        ["uv", "pip", "freeze", "-p", str(clean_install)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "pandas" not in freeze.lower(), "pandas became a runtime dependency"


def test_import_after_clean_install(clean_install):
    proc = subprocess.run(
        [str(clean_install), "-c", "import behaviorgpt; behaviorgpt.UnboxAIClient"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_to_pandas_explains_a_missing_pandas(clean_install):
    code = (
        "from behaviorgpt.types import UnboxAIResponse\n"
        "r = UnboxAIResponse(items=[], offset=0, limit=10)\n"
        "try:\n    r.to_pandas()\n"
        "except ImportError as e:\n    print(e)\n"
    )
    proc = subprocess.run(
        [str(clean_install), "-c", code], capture_output=True, text=True
    )
    assert "pip install pandas" in proc.stdout, proc.stderr[-2000:]
