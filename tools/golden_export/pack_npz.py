#!/usr/bin/env python3
"""Packs each golden-fixture directory of .npy arrays into a single .npz.

The C++ exporter writes one .npy per array because emitting a zip container
from C++ would mean hand-rolling CRC32 and the zip local-file header for no
benefit. One .npz per fixture is what actually gets copied to a rented GPU box,
so the packing happens here instead.

Usage: pack_npz.py [fixture_root]   (default: fixtures)
"""

import pathlib
import sys

import numpy as np


def main() -> int:
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures")
    if not root.is_dir():
        print(f"no fixture root at {root}", file=sys.stderr)
        return 1

    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if not directories:
        print(f"no fixture directories under {root}", file=sys.stderr)
        return 1

    for directory in directories:
        arrays = {path.stem: np.load(path) for path in sorted(directory.glob("*.npy"))}
        if not arrays:
            print(f"{directory} holds no .npy files", file=sys.stderr)
            return 1

        target = root / f"{directory.name}.npz"
        np.savez(target, **arrays)

        # A fixture that does not reload is worse than no fixture, since the GPU
        # kernel would be validated against whatever did survive the round trip.
        with np.load(target) as reloaded:
            for name, array in arrays.items():
                if not np.array_equal(reloaded[name], array):
                    print(f"{target}: {name} changed in the round trip", file=sys.stderr)
                    return 1

        size_kib = target.stat().st_size / 1024
        print(f"{target}  ({len(arrays)} arrays, {size_kib:.1f} KiB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
