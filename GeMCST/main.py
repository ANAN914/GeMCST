"""Unified command-line entry point for GeMCST preprocessing and evaluation.

Maintainer: hanyiyu
Updated: 2026-10-29
"""

from __future__ import annotations

import argparse
import os
import sys


STAGES = {
    "image": ("run_image", "run_graph"),
    "train": ("run_train",),
    "evaluate": ("run_evaluate",),
    "all": ("run_image", "run_graph", "run_train", "run_evaluate"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the GeMCST spatial-domain pipeline.")
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--metadata-path")
    parser.add_argument("--dataset-profile", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-domains-mode", default="auto")
    args = parser.parse_args(argv)

    os.environ.update(
        DATA_DIR=args.data_dir,
        SAMPLE_ID=args.sample_id,
        RESULTS_DIR=args.results_dir,
        DATASET_PROFILE=args.dataset_profile,
        SEED=str(args.seed),
        N_DOMAINS_MODE=args.n_domains_mode,
    )
    if args.metadata_path:
        os.environ["METADATA_PATH"] = args.metadata_path
    else:
        os.environ.pop("METADATA_PATH", None)

    from . import project
    original_argv = sys.argv
    try:
        for stage_name in STAGES[args.stage]:
            sys.argv = [stage_name]
            print(f"Running {stage_name}", flush=True)
            try:
                getattr(project, stage_name)()
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    return int(exc.code) if isinstance(exc.code, int) else 1
    finally:
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
