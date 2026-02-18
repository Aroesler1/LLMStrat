from pathlib import Path
import platform

from quantaalpha.core.experiment import FBWorkspace


def test_link_all_files_in_folder_to_workspace_supports_current_platform(tmp_path: Path):
    data_dir = tmp_path / "data"
    ws_dir = tmp_path / "ws"
    data_dir.mkdir()
    ws_dir.mkdir()

    src = data_dir / "daily_pv.h5"
    src.write_text("sample")

    FBWorkspace.link_all_files_in_folder_to_workspace(data_dir, ws_dir)

    dst = ws_dir / "daily_pv.h5"
    assert dst.exists()
    assert dst.read_text() == "sample"

    if platform.system() in {"Linux", "Darwin"}:
        assert dst.is_symlink()
