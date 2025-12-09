from __future__ import annotations

import argparse
import logging
import os
import sys

# Make this file usable both as a module (`-m ptp_discovery.run_free_loss_eoh`)
# and as a direct script (`python ptp_discovery/run_free_loss_eoh.py`).
if __package__ is None or __package__ == "":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from ptp_discovery.free_loss_eoh_loop import run_free_loss_eoh


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EoH-style discovery of free-form preference losses.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config file for free loss discovery.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device string (e.g., cuda or cpu).",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    )

    parser = _build_arg_parser()
    args = parser.parse_args()

    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device

    run_free_loss_eoh(args.config, **overrides)


if __name__ == "__main__":
    main()
