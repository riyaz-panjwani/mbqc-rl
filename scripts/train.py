"""
Training entry point.

Usage
-----
# Sanity check — 2×2 grid, no defects, 20k steps (fast, ~30 s on CPU)
python scripts/train.py --rows 2 --cols 2 --total-timesteps 20000

# Phase-2 milestone — 3×3 grid, no defects (expect reward → ~1.0)
python scripts/train.py --rows 3 --cols 3 --total-timesteps 200000

# Phase-3 defect curriculum — start training at 20% defect rate
python scripts/train.py --rows 3 --cols 3 --defect-rate 0.2 --total-timesteps 500000

# Load a checkpoint and continue training
python scripts/train.py --rows 3 --cols 3 --resume checkpoints/ckpt_update00050.pt
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

# Allow running as `python scripts/train.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env   import MBQCEnv
from mbqc_rl.agent.policy   import MBQCActorCritic
from mbqc_rl.agent.ppo      import PPOTrainer

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PPO agent on MBQCEnv")

    # Environment
    p.add_argument("--rows",        type=int,   default=3,       help="Grid rows")
    p.add_argument("--cols",        type=int,   default=3,       help="Grid cols (≥2)")
    p.add_argument("--defect-rate", type=float, default=0.0,     help="Edge deletion probability")

    # Training scale
    p.add_argument("--total-timesteps", type=int,   default=200_000)
    p.add_argument("--n-steps",         type=int,   default=512,   help="Steps per rollout")
    p.add_argument("--n-epochs",        type=int,   default=10,    help="PPO update epochs per rollout")
    p.add_argument("--batch-size",      type=int,   default=64,    help="Mini-batch size")

    # PPO hyperparameters
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--gae-lambda",  type=float, default=0.95)
    p.add_argument("--clip-range",  type=float, default=0.2)
    p.add_argument("--vf-coef",     type=float, default=0.5)
    p.add_argument("--ent-coef",    type=float, default=0.01)

    # Network
    p.add_argument("--hidden-dim",  type=int,   default=256)

    # Misc
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--save-dir",     type=str,   default="checkpoints")
    p.add_argument("--log-interval", type=int,   default=10)
    p.add_argument("--save-interval",type=int,   default=50)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to a checkpoint to resume from")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env     = MBQCEnv(rows=args.rows, cols=args.cols,
                      defect_rate=args.defect_rate, seed=args.seed)
    n       = env.n
    obs_dim = n * n + n

    policy    = MBQCActorCritic(obs_dim=obs_dim, n_actions=n, hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    start_update = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        policy.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = ckpt["update"]
        print(f"Resumed from {args.resume} (update {start_update})")

    trainer = PPOTrainer(
        env         = env,
        policy      = policy,
        optimizer   = optimizer,
        n_steps     = args.n_steps,
        n_epochs    = args.n_epochs,
        batch_size  = args.batch_size,
        gamma       = args.gamma,
        gae_lambda  = args.gae_lambda,
        clip_range  = args.clip_range,
        vf_coef     = args.vf_coef,
        ent_coef    = args.ent_coef,
    )

    history = trainer.train(
        total_timesteps = args.total_timesteps,
        save_dir        = args.save_dir,
        log_interval    = args.log_interval,
        save_interval   = args.save_interval,
    )

    # Save final model
    final_path = os.path.join(args.save_dir, "policy_final.pt")
    torch.save({
        "policy_state_dict":    policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history":              history,
        "args":                 vars(args),
    }, final_path)
    print(f"Final model saved → {final_path}")


if __name__ == "__main__":
    main()
