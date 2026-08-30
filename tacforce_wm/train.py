from __future__ import annotations

import argparse

from utils.train_utils import build_canonical_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TacForceWM training entry point")
    subparsers = parser.add_subparsers(dest="task", required=False)

    parser.set_defaults(
        task="dynamics",
        config="tacforce_wm/config/dynamics_train.yaml",
        ddp=False,
        test=False,
    )

    policy = subparsers.add_parser("policy", help="Train flow-matching policy")
    policy.add_argument("--config", type=str, default="config/dynamics_train.yaml")
    policy.add_argument(
        "--benchmark",
        type=str,
        default="none",
        choices=["none", "dataloader", "model", "both"],
    )
    policy.add_argument("--bench-steps", type=int, default=100)
    policy.add_argument("--bench-warmup", type=int, default=10)

    dynamics = subparsers.add_parser("dynamics", help="Train TacForce world model")
    dynamics.add_argument("--config", type=str, default="config/dynamics_train.yaml")
    dynamics.add_argument("--ddp", action="store_true")
    dynamics.add_argument("--test", action="store_true")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.task == "policy":
        overrides = {
            "runtime.benchmark": getattr(args, "benchmark", "none"),
            "runtime.bench_steps": getattr(args, "bench_steps", 100),
            "runtime.bench_warmup": getattr(args, "bench_warmup", 10),
        }
        cfg = build_canonical_config("policy", args.config, overrides=overrides)
        from trainers.policy_trainer import main as policy_main

        policy_main(cfg)
        return

    if args.task == "dynamics":
        overrides = {
            "runtime.ddp": getattr(args, "ddp", False),
            "runtime.test": getattr(args, "test", False),
        }
        cfg = build_canonical_config("dynamics", args.config, overrides=overrides)
        from trainers.dynamics_trainer import main as dynamics_main

        dynamics_main(cfg)
        return

    raise ValueError(f"Unsupported task: {args.task}")


if __name__ == "__main__":
    main()
