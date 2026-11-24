"""Tests de la fusion des réglages : défauts < gestures.yaml < local.yaml < ligne de commande."""

import yaml

from airdesk.config import (
    CameraConfig,
    ZoneConfig,
    apply_cli_overrides,
    load_config,
    merge_settings,
    read_yaml,
    save_camera_settings,
    save_local_section,
    save_zone,
)


def write_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return path


# --- merge_settings ---------------------------------------------------------------------------


def test_merge_overrides_only_the_given_keys():
    base = {"pinch": {"enter": 0.25, "exit": 0.40}}
    override = {"pinch": {"enter": 0.20}}
    assert merge_settings(base, override) == {"pinch": {"enter": 0.20, "exit": 0.40}}


def test_merge_adds_a_section_absent_from_base():
    merged = merge_settings({"pinch": {"enter": 0.25}}, {"camera": {"index": 1}})
    assert merged == {"pinch": {"enter": 0.25}, "camera": {"index": 1}}


def test_merge_does_not_modify_its_inputs():
    base = {"pinch": {"enter": 0.25}}
    override = {"pinch": {"enter": 0.20}}
    merge_settings(base, override)
    assert base == {"pinch": {"enter": 0.25}}


def test_merge_accepts_an_empty_section():
    """En YAML, « camera: » sans rien dessous donne None : ça ne doit pas planter."""
    assert merge_settings({"camera": None}, {"camera": {"index": 2}}) == {"camera": {"index": 2}}


# --- load_config ------------------------------------------------------------------------------


def test_defaults_when_no_file_exists(tmp_path):
    cfg = load_config(tmp_path / "absent.yaml", local_path=tmp_path / "absent_local.yaml")
    assert cfg.camera == CameraConfig()
    assert cfg.pinch.click_max_ms == 300


def test_local_file_overrides_gestures_file(tmp_path):
    gestures = write_yaml(tmp_path / "gestures.yaml", {"pinch": {"enter": 0.25, "exit": 0.40}})
    local = write_yaml(
        tmp_path / "local.yaml",
        {"pinch": {"enter": 0.20}, "camera": {"index": 1, "backend": "dshow"}},
    )
    cfg = load_config(gestures, local_path=local)
    assert cfg.pinch.enter == 0.20  # surchargé par local.yaml
    assert cfg.pinch.exit == 0.40  # conservé depuis gestures.yaml
    assert cfg.camera.index == 1
    assert cfg.camera.backend == "dshow"
    assert cfg.camera.width == 1280  # valeur par défaut


def test_local_file_can_be_ignored(tmp_path):
    local = write_yaml(tmp_path / "local.yaml", {"camera": {"index": 3}})
    cfg = load_config(tmp_path / "absent.yaml", local_path=None)
    assert cfg.camera.index == 0
    assert local.exists()


# --- Ligne de commande ------------------------------------------------------------------------


def test_cli_options_override_files():
    from_files = CameraConfig(index=1, backend="dshow", width=640, height=480)
    result = apply_cli_overrides(from_files, index=2, width=1280)
    assert result == CameraConfig(index=2, backend="dshow", width=1280, height=480)


def test_cli_options_not_given_keep_file_values():
    from_files = CameraConfig(index=1, backend="dshow", width=640, height=480)
    assert apply_cli_overrides(from_files) == from_files


def test_full_priority_chain(tmp_path):
    """Défaut 1280x720 < local.yaml (index 1, 640x480) < ligne de commande (--camera 0)."""
    local = write_yaml(
        tmp_path / "local.yaml", {"camera": {"index": 1, "width": 640, "height": 480}}
    )
    cfg = load_config(tmp_path / "absent.yaml", local_path=local)
    camera = apply_cli_overrides(cfg.camera, index=0)
    assert camera == CameraConfig(index=0, backend="auto", width=640, height=480)


# --- save_camera_settings ---------------------------------------------------------------------


def test_save_creates_the_local_file(tmp_path):
    path = tmp_path / "config" / "local.yaml"
    save_camera_settings(path, 1, "dshow", 640, 480)
    cfg = load_config(tmp_path / "absent.yaml", local_path=path)
    assert cfg.camera == CameraConfig(index=1, backend="dshow", width=640, height=480)


def test_save_zone_keeps_the_other_pointer_keys(tmp_path):
    path = write_yaml(tmp_path / "local.yaml", {"pointer": {"mode": "relative"},
                                                "camera": {"index": 1}})
    save_zone(path, ZoneConfig(left=0.1, right=0.8, top=0.3, bottom=0.9))
    raw = read_yaml(path)
    assert raw["pointer"]["mode"] == "relative"  # conservé
    assert raw["pointer"]["zone"] == {"left": 0.1, "right": 0.8, "top": 0.3, "bottom": 0.9}
    assert raw["camera"] == {"index": 1}
    cfg = load_config(tmp_path / "absent.yaml", local_path=path)
    assert cfg.pointer.zone == ZoneConfig(left=0.1, right=0.8, top=0.3, bottom=0.9)


def test_save_local_section_merges_keys(tmp_path):
    path = write_yaml(tmp_path / "local.yaml", {"pinch": {"enter": 0.2, "exit": 0.4}})
    save_local_section(path, "pinch", {"enter": 0.22})
    assert read_yaml(path)["pinch"] == {"enter": 0.22, "exit": 0.4}


def test_save_keeps_the_other_sections(tmp_path):
    path = write_yaml(
        tmp_path / "local.yaml", {"pinch": {"enter": 0.2}, "camera": {"index": 0}}
    )
    save_camera_settings(path, 1, "dshow", 640, 480)
    raw = read_yaml(path)
    assert raw["pinch"] == {"enter": 0.2}
    assert raw["camera"] == {"index": 1, "backend": "dshow", "width": 640, "height": 480}
