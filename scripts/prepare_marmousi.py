#!/usr/bin/env python3
"""Download and prepare the true elastic Marmousi-II benchmark fields."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridadfwi.models import PRESETS, prepare_marmousi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/marmousi2")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smooth-sigma-m", type=float, default=150.0)
    parser.add_argument("--density-mode", choices=("constant2450", "source"), default="constant2450")
    args = parser.parse_args()
    output = args.output or ROOT / f"data/prepared/marmousi2_{args.preset}.npz"
    result = prepare_marmousi(output, args.raw_dir, args.preset, args.smooth_sigma_m, args.density_mode)
    print(result)
    print(result.with_suffix(".json"))


if __name__ == "__main__":
    main()
