from pathlib import Path

import pytest

from humanoid_vla.sim.assets import MENAGERIE_COMMIT, AssetError, g1_scene_xml, menagerie_head


def test_menagerie_is_at_pinned_commit() -> None:
    assert menagerie_head() == MENAGERIE_COMMIT
    assert g1_scene_xml().is_file()


def test_missing_checkout_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AssetError, match="make assets"):
        g1_scene_xml(tmp_path)


def test_checkout_at_other_commit_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("0" * 40 + "\n")
    with pytest.raises(AssetError, match="expected pinned commit"):
        g1_scene_xml(tmp_path)
