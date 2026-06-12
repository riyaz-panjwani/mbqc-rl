"""
Mixed-rate training — the anti-catastrophic-forgetting alternative
to the staged curriculum.

Motivation (Session 5 finding)
------------------------------
The staged curriculum (0% → 10% → 20% → 30%) caused catastrophic
forgetting on 4×4 and 5×5 grids: Stage-0 mastered clean graphs
(reward ≈ 0.99) but Stage-3 lost that skill (4×4: 0.75, 5×5: 0.51 at
1% defects, vs Stage-0 checkpoints at 0.95/0.93).

The fix tested here: instead of training in stages, each EPISODE
samples a defect rate uniformly from [0, 0.3]. The agent sees all
regimes interleaved throughout training, so no regime is forgotten.
Same total timestep budget as the staged curriculum for a fair
comparison.

Usage
-----
# 4×4 mixed training, same budget as the staged curriculum (1.5M steps)
python scripts/train_mixed.py --rows 4 --cols 4 \\
    --total-timesteps 1500000 --save-dir results/4x4_mixed

# 3×3 with non-Clifford measurement angles (universal MBQC)
python scripts/train_mixed.py --rows 3 --cols 3 --use-angles \\
    --total-timesteps 750000 --save-dir results/3x3_angled_mixed
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env   import MBQCEnv
from mbqc_rl.agent.policy   import MBQCActorCritic
from mbqc_rl.agent.ppo      import PPOTrainer
from mbqc_rl.utils.metrics  import save_history, summarise, plot_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mixed defect-rate PPO training")
    p.add_argument("--rows",        type=int,   default=3)
    p.add_argument("--cols",        type=int,   default=3)
    p.add_argument("--defect-lo",   type=float, default=0.0,
                   help="Lower bound of per-episode defect rate")
    p.add_argument("--defect-hi",   type=float, default=0.3,
                   help="Upper bound of per-episode defect rate")
    p.add_argument("--use-angles",  action="store_true",
                   help="Universal MBQC: random k·π/4 angles, angle-aware reward")
    p.add_argument("--clifford-fraction", type=float, default=0.5)
    p.add_argument("--total-timesteps", type=int, default=750_000)
    p.add_argument("--save-dir",    type=str,   default="results/mixed")
    # PPO hyperparameters (same defaults as curriculum.py)
    p.add_argument("--n-steps",     type=int,   default=512)
    p.add_argument("--n-epochs",    type=int,   default=10)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--gamma",       type=float, default=0.99)
    p.add_argument("--gae-lambda",  type=float, default=0.95)
    p.add_argument("--clip-range",  type=float, default=0.2)
    p.add_argument("--vf-coef",     type=float, default=0.5)
    p.add_argument("--ent-coef",    type=float, default=0.01)
    p.add_argument("--hidden-dim",  type=int,   default=256)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log-interval", type=int,  default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    n       = args.rows * args.cols
    obs_dim = n * n + n + (n if args.use_angles else 0)

    env = MBQCEnv(
        rows=args.rows, cols=args.cols,
        defect_rate=(args.defect_lo, args.defect_hi),   # ← per-episode sampling
        use_angles=args.use_angles,
        clifford_fraction=args.clifford_fraction,
        seed=args.seed,
    )

    policy    = MBQCActorCritic(obs_dim=obs_dim, n_actions=n,
                                hidden_dim=args.hidden_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

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

    print(f"\n{'═'*60}")
    print(f"  MIXED-RATE TRAINING: {args.rows}×{args.cols} grid")
    print(f"  defect_rate ~ U[{args.defect_lo}, {args.defect_hi}] per episode")
    print(f"  use_angles={args.use_angles}  "
          f"clifford_fraction={args.clifford_fraction}")
    print(f"  total timesteps: {args.total_timesteps:,}")
    print(f"{'═'*60}")

    history = trainer.train(
        total_timesteps = args.total_timesteps,
        save_dir        = os.path.join(args.save_dir, "checkpoints"),
        log_interval    = args.log_interval,
        save_interval   = 50,
    )

    # Save final checkpoint with full config
    ckpt_path = os.path.join(args.save_dir, "final.pt")
    torch.save({
        "policy_state_dict":    policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(rows=args.rows, cols=args.cols,
                       obs_dim=obs_dim, n_actions=n,
                       hidden_dim=args.hidden_dim,
                       use_angles=args.use_angles,
                       clifford_fraction=args.clifford_fraction,
                       defect_lo=args.defect_lo, defect_hi=args.defect_hi),
    }, ckpt_path)
    print(f"\n  Final checkpoint → {ckpt_path}")

    save_history(history, os.path.join(args.save_dir, "history.json"))
    summarise(history)
    plot_training(
        history,
        save_to = os.path.join(args.save_dir, "training_curve.png"),
        title   = (f"Mixed-Rate Training — {args.rows}×{args.cols} "
                   f"U[{args.defect_lo},{args.defect_hi}]"
                   + (" + angles" if args.use_angles else "")),
        show    = False,
    )


if __name__ == "__main__":
    main()
