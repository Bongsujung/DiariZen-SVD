#!/usr/bin/env python3
"""Flatten conf/run.toml into shell variables for run_stage.sh:  eval "$(python scripts/_config.py conf/run.toml)".

[paths] exp = "exp"   ->   PATHS_EXP='exp'
lists are joined with spaces so that bash can iterate over them:  ratios = [5, 2]  ->  COMPRESS_RATIOS='5 2'
"""
import shlex
import sys

import toml


def main():
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__); return
    cfg = toml.load(sys.argv[1] if len(sys.argv) > 1 else "conf/run.toml")
    for section, entries in cfg.items():
        for key, value in entries.items():
            if isinstance(value, list):
                value = " ".join(str(v) for v in value)
            print(f"{section.upper()}_{key.upper()}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
