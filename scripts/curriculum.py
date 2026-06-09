"""
Defect-rate curriculum training — the domain-randomisation strategy
described in the dissertation proposal (after Tobin et al., 2017).

Strategy
--------
Train the policy through a sequence of stages with increasing defect
rates. Each stage loads the previous stage's checkpoint and continues
training on a harder distribution. By the end, the policy has been
exposed to defect rates from 0% to 30%, making it conservative and
robust. At evaluation/deployment time it is tested at the realistic 1%.

Why this works (Tobin et al., 2017 analogy)
--------------------------------------------
A policy trained only on easy (0% defect) instances learns to follow
the trivial gflow order and overfits to perfect graphs. Exposing it
progressively to broken graphs forces it to develop robust heuristics
for navigating missing edges — skills it then retains even on near-perfect
deployment graphs.

Default curriculum (3×3 grid)
------------------------------
  Stage 0: defect_rate=0.00, 200k steps   ← learn basic ordering
  Stage 1: defect_rate=0.10, 150k steps   ← mild defects
  Stage 2: defect_rate=0.20, 200k steps   ← moderate (realistic training)
  Stage 3: defect_rate=0.30, 200k steps   ← severe (curriculum peak)

Usage
-----
# Full curriculum on 3×3 grid
python scripts/curriculum.py --rows 3 --cols 3 --save-dir results/3x3

# Quick test run (~1 min) — 20k steps per stage
python scripts/curriculum.py --rows 3 --cols 3 --quick --save-dir results/3x3_quick

# Resume from a specific stage checkpoint
python scripts/curriculum.py --rows 3 --cols 3 --resume-stage 2 \\
    --resume-ckpt results/3x3/stage1_final.pt --save-dir results/3x3
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env import MBQCEnv
from mbqc_rl.agent.policy import MBQCActorCritic
from mbqc_rl.agent.ppo    import PPOTrainer
from mbqc_rl.utils.metrics import save_history, summarise, plot_training


# Default curriculum — can override per stage with --timesteps-per-stage
DEFAULT_CURRICULUM = [
    dict(defect_rate=0.00, timesteps=200_000, label="Stage 0 — no defects"),
    dict(defect_rate=0.10, timesteps=150_000, label="Stage 1 — 10% defects"),
    dict(defect_rate=0.20, timesteps=200_000, label="Stage 2 — 20% defects"),
    dict(defect_rate=0.30, timesteps=200_000, label="Stage 3 — 30% defects"),
]

QUICK_CURRICULUM = [
    dict(defect_rate=0.00, timesteps=20_000, label="Stage 0 — no defects (quick)"),
    dict(defect_rate=0.10, timesteps=15_000, label="Stage 1 — 10% defects (quick)"),
    dict(defect_rate=0.20, timesteps=20_000, label="Stage 2 — 20% defects (quick)"),
    dict(defect_rate=0.30, timesteps=20_000, label="Stage 3 — 30% defects (quick)"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Defect-rate curriculum PPO training")
    p.add_argument("--rows",       type=int,   default=3)
    p.add_argument("--cols",       type=int,   default=3)
    p.add_argument("--save-dir",   type=str,   default="results/curriculum")
    p.add_argument("--quick",      action="store_true",
                   help="Run a short version of the curriculum (~1 min per stage)")
    # PPO hyperparameters
    p.add_argument("--n-steps",    type=int,   default=512)
    p.add_argument("--n-epochs",   type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--gamma",      type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--vf-coef",    type=float, default=0.5)
    p.add_argument("--ent-coef",   type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int,   default=256)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--log-interval", type=int,   default=10)
    # Resume
    p.add_argument("--resume-stage", type=int, default=0,
                   help="Index (0-based) of the first stage to run")
    p.add_argument("--resume-ckpt",  type=str, default=None,
                   help="Checkpoint to load before the resume stage")
    return p.parse_args()


def make_policy(rows: int, cols: int, hidden_dim: int) -> MBQCActorCritic:
    n       = rows * cols
    obs_dim = n * n + n
    return MBQCActorCritic(obs_dim=obs_dim, n_actions=n, hidden_dim=hidden_dim)


def main() -> None:
    args       = parse_args()
    curriculum = QUICK_CURRICULUM if args.quick else DEFAULT_CURRICULUM

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    n          = args.rows * args.cols
    obs_dim    = n * n + n
    policy     = make_policy(args.rows, args.cols, args.hidden_dim)
    optimizer  = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    all_history: list[dict] = []

    # Optionally resume from a checkpoint into a specific stage
    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu")
        policy.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Loaded checkpoint: {args.resume_ckpt}")

    print(f"\n{'═'*60}")
    print(f"  CURRICULUM: {args.rows}×{args.cols} grid  "
          f"({'quick' if args.quick else 'full'})")
    print(f"  {len(curriculum)} stages  |  save_dir: {args.save_dir}")
    print(f"{'═'*60}")

    for stage_idx, stage in enumerate(curriculum):
        if stage_idx < args.resume_stage:
            print(f"  Skipping {stage['label']}")
            continue

        print(f"\n  ▶  {stage['label']}  "
              f"({stage['timesteps']:,} timesteps)")

        env = MBQCEnv(
            rows=args.rows, cols=args.cols,
            defect_rate=stage["defect_rate"],
            seed=args.seed + stage_idx,
        )

        trainer = PPOTrainer(
            env=env, policy=policy, optimizer=optimizer,
            n_steps    = args.n_steps,
            n_epochs   = args.n_epochs,
            batch_size = args.batch_size,
            gamma      = args.gamma,
            gae_lambda = args.gae_lambda,
            clip_range = args.clip_range,
            vf_coef    = args.vf_coef,
            ent_coef   = args.ent_coef,
        )
        # Carry over running env state between stages isn't meaningful
        # (different defect rates), so force a reset:
        trainer._needs_reset = True

        stage_dir = os.path.join(args.save_dir, f"stage{stage_idx}")
        history = trainer.train(
            total_timesteps = stage["timesteps"],
            save_dir        = stage_dir,
            log_interval    = args.log_interval,
            save_interval   = 50,
        )

        # Tag each record with stage info
        for record in history:
            record["stage"]       = stage_idx
            record["defect_rate"] = stage["defect_rate"]
        all_history.extend(history)

        # Save stage checkpoint and history
        ckpt_path = os.path.join(args.save_dir, f"stage{stage_idx}_final.pt")
        torch.save({
            "stage":                stage_idx,
            "defect_rate":          stage["defect_rate"],
            "policy_state_dict":    policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dict(rows=args.rows, cols=args.cols,
                           obs_dim=obs_dim, n_actions=n,
                           hidden_dim=args.hidden_dim),
        }, ckpt_path)
        print(f"  Stage {stage_idx} checkpoint → {ckpt_path}")

        hist_path = os.path.join(args.save_dir, f"stage{stage_idx}_history.json")
        save_history(history, hist_path)
        summarise(history)

    # Save full curriculum history
    full_path = os.path.join(args.save_dir, "full_curriculum_history.json")
    save_history(all_history, full_path)
    print(f"\nFull history → {full_path}")

    # Plot
    plot_path = os.path.join(args.save_dir, "curriculum_training_curve.png")
    plot_training(
        all_history,
        save_to  = plot_path,
        title    = f"Curriculum Training — {args.rows}×{args.cols} grid",
        show     = False,
    )
    print(f"Training curve → {plot_path}")


if __name__ == "__main__":
    main()
