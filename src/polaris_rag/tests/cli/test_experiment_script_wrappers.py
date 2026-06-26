from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "experiments"
SRC_DIR = REPO_ROOT / "src"


def _install_stub_cli_module(monkeypatch, module_name: str, calls: list[dict[str, str]]) -> None:
    pkg = ModuleType("polaris_rag")
    pkg.__path__ = []
    cli_pkg = ModuleType("polaris_rag.cli")
    cli_pkg.__path__ = []
    module = ModuleType(module_name)

    def _main() -> None:
        calls.append(
            {
                "module": module_name,
                "sys_path0": sys.path[0],
                "argv0": sys.argv[0],
            }
        )

    module.main = _main
    setattr(cli_pkg, module_name.rsplit(".", 1)[-1], module)
    setattr(pkg, "cli", cli_pkg)

    monkeypatch.setitem(sys.modules, "polaris_rag", pkg)
    monkeypatch.setitem(sys.modules, "polaris_rag.cli", cli_pkg)
    monkeypatch.setitem(sys.modules, module_name, module)


@pytest.mark.parametrize(
    ("script_name", "module_name"),
    [
        ("render_config.py", "polaris_rag.cli.render_experiment_config"),
        ("run_stage.py", "polaris_rag.cli.run_experiment_stage"),
        ("summarize_stage.py", "polaris_rag.cli.summarize_experiment_stage"),
    ],
)
def test_experiment_script_wrappers_bootstrap_src_and_delegate(
    monkeypatch,
    script_name: str,
    module_name: str,
) -> None:
    calls: list[dict[str, str]] = []
    _install_stub_cli_module(monkeypatch, module_name, calls)

    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path if entry != str(SRC_DIR)])
    monkeypatch.setattr(sys, "argv", [script_name])

    runpy.run_path(str(SCRIPTS_DIR / script_name), run_name="__main__")

    assert calls == [
        {
            "module": module_name,
            "sys_path0": str(SRC_DIR),
            "argv0": str(SCRIPTS_DIR / script_name),
        }
    ]
    assert sys.path.count(str(SRC_DIR)) == 1
