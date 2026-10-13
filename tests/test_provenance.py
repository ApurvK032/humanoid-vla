import json
import re
from pathlib import Path

from humanoid_vla.utils.provenance import provenance, write_result


def test_provenance_identifies_code_and_environment() -> None:
    started = provenance()
    assert re.fullmatch(r"[0-9a-f]{40}", started["git"]["commit"])
    assert isinstance(started["git"]["dirty"], bool)
    assert started["packages"]["mujoco"] is not None
    assert started["python"].startswith("3.12")


def test_write_result_stores_provenance_from_run_start(tmp_path: Path) -> None:
    started = provenance()
    path = tmp_path / "nested" / "result.json"
    write_result(path, {"metric": 1.5}, started)
    saved = json.loads(path.read_text())
    assert saved == {"provenance": started, "metric": 1.5}
