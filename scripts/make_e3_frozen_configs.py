"""
Derive the E3 frozen configs: the H-V and Q-V frozen configs
with only `batch_size` changed, one subdirectory per batch size.

    python scripts/make_e3_frozen_configs.py \
        --source_dir configs/sgd_frozen --out_dir configs/sgd_frozen_e3 \
        --batch_sizes 16,32,64,128,256

writes configs/sgd_frozen_e3/b{b}/h_v_frozen_config.json and
q_v_frozen_config.json. Every other field (learning rate, network, kernel
multiplier, confirm_status, ...) is copied unchanged; config_hash is
recomputed by FrozenConfig.save(). Existing files are never overwritten
unless --overwrite is given.
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hpo_protocol import FrozenConfig

E3_CONFIG_FILES = ("h_v_frozen_config.json", "q_v_frozen_config.json")


def make_e3_configs(source_dir: Path, out_dir: Path, batch_sizes: List[int], overwrite: bool = False) -> Dict[int, List[Path]]:
    """Write one H-V/Q-V pair per batch size; returns {batch_size: [written paths]}."""
    if not batch_sizes:
        raise ValueError("batch_sizes is empty.")
    for b in batch_sizes:
        if int(b) < 2:
            raise ValueError(f"batch_size must be >= 2, got {b}.")
    sources = {name: FrozenConfig.load(Path(source_dir) / name) for name in E3_CONFIG_FILES}
    for name, cfg in sources.items():
        if cfg.training_objective != "V":
            raise ValueError(f"{name} has training_objective={cfg.training_objective!r}; E3 derives from V configs only.")
    written: Dict[int, List[Path]] = {}
    for b in batch_sizes:
        sub = Path(out_dir) / f"b{int(b)}"
        paths = []
        for name, cfg in sources.items():
            path = sub / name
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
            replace(cfg, batch_size=int(b)).save(path)
            paths.append(path)
        written[int(b)] = paths
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source_dir", default="configs/sgd_frozen")
    parser.add_argument("--out_dir", default="configs/sgd_frozen_e3")
    parser.add_argument("--batch_sizes", default="16,32,64,128,256")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    written = make_e3_configs(Path(args.source_dir), Path(args.out_dir), sizes, overwrite=args.overwrite)
    for b, paths in written.items():
        for p in paths:
            print(f"b={b}: {p}")


if __name__ == "__main__":
    main()
