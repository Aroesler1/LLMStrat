from __future__ import annotations

import sys
from pathlib import Path


def us_root_from_script(script_file: str) -> Path:
    return Path(script_file).resolve().parents[1]


def bootstrap_us_paths(script_file: str) -> Path:
    """Ensure `quantaalpha_us` is importable when running scripts directly."""
    us_root = us_root_from_script(script_file)
    for p in (str(us_root), str(us_root.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return us_root


def resolve_from_us_root(path_value: str | Path, us_root: Path) -> Path:
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p
    return (us_root / p).resolve()
