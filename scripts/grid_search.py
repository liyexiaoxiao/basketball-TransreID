"""
Hyperparameter grid search config generator.

Generates YAML config files for each (key, value) combination,
based on the baseline config vit_all_optimizations.yml.
Each variant overrides one hyperparameter via CLI-compatible output.

Usage:
    python scripts/grid_search.py --dry-run    # Print commands only
    python scripts/grid_search.py              # Generate config files in configs/BallShow/grid/
"""

import os
import argparse
from pathlib import Path

BASE_CONFIG = 'configs/BallShow/vit_all_optimizations.yml'
GRID_DIR = 'configs/BallShow/grid'

# Grid search space
GRID = {
    # ArcFace angular margin
    'COSINE_MARGIN': [0.3, 0.4, 0.5, 0.6],
    # ArcFace scale
    'COSINE_SCALE': [15, 30, 45],
    # Circle Loss scale (SOLVER)
    'CIRCLE_S': [64, 128, 256],
    # Circle Loss margin (SOLVER)
    'CIRCLE_M': [0.15, 0.25, 0.35],
    # AdamW learning rate (SOLVER)
    'BASE_LR': [0.0001, 0.00035, 0.0007],
    # AdamW weight decay (SOLVER)
    'WEIGHT_DECAY': [0.01, 0.05, 0.1],
    # Patch drop probability (MODEL)
    'PATCH_DROP_PROB': [0.0, 0.1, 0.2, 0.3],
    # MixStyle probability (MODEL)
    'MIXSTYLE_P': [0.0, 0.2, 0.3, 0.5],
    # Projection dimension (MODEL) — disabled: accuracy > compression
    # 'PROJ_DIM': [0, 128, 256, 512],
}

# Recommended focused search (most impactful params only)
FOCUSED_GRID = {
    'COSINE_MARGIN': [0.4, 0.5, 0.6],
    'CIRCLE_S': [64, 128, 256],
    'BASE_LR': [0.0001, 0.00035, 0.0007],
    'PATCH_DROP_PROB': [0.1, 0.2, 0.3],
}


# Config section prefixes for each parameter
PARAM_PREFIX = {
    'COSINE_MARGIN': 'SOLVER',
    'COSINE_SCALE': 'SOLVER',
    'CIRCLE_S': 'SOLVER',
    'CIRCLE_M': 'SOLVER',
    'BASE_LR': 'SOLVER',
    'WEIGHT_DECAY': 'SOLVER',
    'PATCH_DROP_PROB': 'MODEL',
    'MIXSTYLE_P': 'MODEL',
    'PROJ_DIM': 'MODEL',
    'DROP_PATH': 'MODEL',
}


def generate_commands(grid, dry_run=True):
    """Generate training commands for each hyperparameter sweep."""
    commands = []

    for param, values in grid.items():
        for val in values:
            if isinstance(val, tuple):
                val_str = str(list(val))
            else:
                val_str = str(val)

            # Build output dir name
            dir_name = f"grid_{param.lower()}_{str(val).replace('.','_').replace(' ','')}"
            output_dir = f"logs/{dir_name}"

            prefix = PARAM_PREFIX.get(param, 'SOLVER')
            cmd = (
                f"python train.py "
                f"--config_file {BASE_CONFIG} "
                f"{prefix}.{param} {val_str} "
                f"OUTPUT_DIR '{output_dir}'"
            )
            commands.append((param, val, cmd))

    return commands


def main():
    parser = argparse.ArgumentParser(description="Grid search config/command generator")
    parser.add_argument('--dry-run', action='store_true', help='Print commands only')
    parser.add_argument('--focused', action='store_true', help='Use focused (smaller) grid')
    parser.add_argument('--param', type=str, default=None,
                        help='Single parameter to sweep (e.g. CIRCLE_SCALE)')
    args = parser.parse_args()

    grid = FOCUSED_GRID if args.focused else GRID
    if args.param:
        if args.param in GRID:
            grid = {args.param: GRID[args.param]}
        else:
            print(f"Unknown parameter: {args.param}")
            print(f"Available: {list(GRID.keys())}")
            return

    commands = generate_commands(grid, dry_run=args.dry_run)

    print(f"=== Grid Search: {len(commands)} experiments ===\n")
    print(f"Base config: {BASE_CONFIG}\n")

    total = 0
    for param, val, cmd in commands:
        total += 1
        print(f"# {param} = {val}")
        print(cmd)
        print()

    print(f"---\nTotal: {total} experiments")
    print(f"\nFull grid: {len(GRID)} params x avg {sum(len(v) for v in GRID.values())//len(GRID)} values = ~{total} runs")
    print(f"Estimated GPU hours @ 6h/run: {total * 6}h on single GPU")


if __name__ == '__main__':
    main()
