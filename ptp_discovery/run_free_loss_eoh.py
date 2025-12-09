from __future__ import annotations

import argparse
import logging

from .free_loss_eoh_loop import run_free_loss_eoh


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
