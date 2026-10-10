"""Configure only the project-owned hf-libero asset/config paths, without home prompts."""

import importlib.metadata
import json
from pathlib import Path

from robot_vla.schema import file_hash


def main():
    root = Path(".").resolve()
    reference = root / "external/LIBERO/libero/libero"
    assert (reference / "assets/scenes/libero_tabletop_base_style.xml").is_file()
    package = Path(
        importlib.metadata.distribution("hf-libero").locate_file("libero/libero")
    ).resolve()
    if not package.is_relative_to(root):
        raise ValueError("Refusing to modify assets outside project-owned environment")
    assets = package / "assets"
    if assets.is_symlink():
        if assets.resolve() != (reference / "assets").resolve():
            raise ValueError("Existing asset link differs; inspect before replacing")
    elif not assets.exists():
        assets.symlink_to(reference / "assets", target_is_directory=True)
    else:
        raise ValueError("Existing non-symlink assets need an explicit provenance audit")
    directory = root / "work/libero"
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(
        benchmark_root=str(reference),
        bddl_files=str(reference / "bddl_files"),
        init_states=str(reference / "init_files"),
        datasets=str(root / "data"),
        assets=str(assets),
    )
    (directory / "config.yaml").write_text(json.dumps(config, indent=2))
    report = dict(
        runtime_distribution="hf-libero",
        version=importlib.metadata.version("hf-libero"),
        runtime_package=str(package),
        asset_reference=str(reference),
        config=config,
        config_sha256=file_hash(directory / "config.yaml"),
    )
    (root / "reports/libero_runtime_provenance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
