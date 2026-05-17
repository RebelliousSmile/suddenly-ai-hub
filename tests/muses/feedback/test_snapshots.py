"""Tests T40 — Snapshots et rollback des stores."""

import time

from muses.feedback.snapshots import (
    list_snapshots,
    restore_snapshot,
    snapshot_directory,
)


def _create_state(d, content="hello"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "trust.sqlite").write_text(content)
    (d / "style.sqlite").write_text("style data")


def test_snapshot_creates_timestamped_copy(tmp_path):
    src = tmp_path / "feedback"
    _create_state(src, "v1")
    snap_dir = tmp_path / "snaps"

    snap = snapshot_directory(src, snap_dir)
    assert snap.exists()
    assert (snap / "trust.sqlite").read_text() == "v1"
    assert snap.parent == snap_dir


def test_list_snapshots_orders_newest_first(tmp_path):
    src = tmp_path / "feedback"
    _create_state(src)
    snap_dir = tmp_path / "snaps"
    snapshot_directory(src, snap_dir)
    time.sleep(1.1)  # avancer d'une seconde pour distinct timestamps
    snapshot_directory(src, snap_dir)
    snaps = list_snapshots(snap_dir)
    assert len(snaps) == 2
    assert snaps[0].name > snaps[1].name


def test_restore_snapshot_overwrites(tmp_path):
    src = tmp_path / "feedback"
    _create_state(src, "v1")
    snap_dir = tmp_path / "snaps"
    snap = snapshot_directory(src, snap_dir)
    # Modifie le state actuel
    (src / "trust.sqlite").write_text("v2")
    # Restaure
    restore_snapshot(snap, src)
    assert (src / "trust.sqlite").read_text() == "v1"


def test_restore_nonexistent_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        restore_snapshot(tmp_path / "absent", tmp_path / "anywhere")
