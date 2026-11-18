# src/config_loader.py
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

PROFILE_ALIASES = {
    "real": "raspberry_pi",
    "rpi": "raspberry_pi",
    "raspberry_pi": "raspberry_pi",
    "sim": "simulation",
    "gazebo": "simulation",
    "simulation": "simulation",
    "mock": "mock",
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Gộp đệ quy 2 dict cấu hình mà không làm thay đổi bản gốc."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Nội dung YAML phải là dictionary: {path}")
    return data


def resolve_project_path(raw_path: str | Path) -> Path:
    """Quy đổi đường dẫn tương đối theo thư mục gốc pbl6_tricopter_vision."""
    path = Path(raw_path)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_config(
    profile: str = "raspberry_pi",
    config_dir: Optional[Path] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Nạp cấu hình theo thứ tự ưu tiên:
    1. common.yaml
    2. <profile>.yaml (hỗ trợ alias real->raspberry_pi, sim->simulation)
    3. overrides (tham số truyền từ CLI hoặc code)
    """
    cfg_dir = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    profile_name = PROFILE_ALIASES.get(profile.lower(), profile.lower())

    common_cfg = _load_yaml_file(cfg_dir / "common.yaml")
    profile_cfg = _load_yaml_file(cfg_dir / f"{profile_name}.yaml")

    merged = deep_merge(common_cfg, profile_cfg)
    if overrides:
        merged = deep_merge(merged, overrides)
    merged["profile"] = profile_name
    return merged


if __name__ == "__main__":
    print("=== Kiểm thử độc lập: config_loader.py ===")
    for p in ("raspberry_pi", "real", "simulation", "sim", "mock"):
        cfg = load_config(p)
        assert "mode" in cfg and "mavlink" in cfg and "detector" in cfg
        print(
            f"[PASS] Profile '{p}' -> '{cfg['profile']}' "
            f"(mode={cfg['mode']}, conn={cfg['mavlink']['conn_str']}, "
            f"baud={cfg['mavlink']['baud']}, web={cfg['web']['enabled']})"
        )
    overridden = load_config("real", overrides={"mavlink": {"baud": 115200}})
    assert overridden["mavlink"]["baud"] == 115200
    print("[PASS] Override từ CLI hoạt động đúng!")
    print("=== Module config_loader.py hoạt động tốt! ===")
