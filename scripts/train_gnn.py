"""
GNN (Graph Attention Network) mixed-rate training.

Identical setup to train_mixed.py but uses GNNActorCritic instead of
the flat MLP.  The GNN processes the adjacency matrix as a real graph,
giving permutation equivariance and (crucially) size-inductive weights:
a policy trained on 3×3 can be benchmarked zero-shot on 4×4 or 5×5.

Usage
-----
# 3×3 baseline GNN (compare with MLP at same budget)
python scripts/train_gnn.py --rows 3 --cols 3 \\
    --total-timesteps 750000 --save-dir results/gnn_3x3

# 3×3 with non-Clifford angles
python scripts/train_gnn.py --rows 3 --cols 3 --use-angles \\
    --total-timesteps 750000 --save-dir results/gnn_3x3_angled

# Transfer-learning evaluation after training:
# Load the 3×3 checkpoint, evaluate on 4×4 — the GNN handles any n.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mbqc_rl.env.mbqc_env        import MBQCEnv
from mbqc_rl.agent.gnn_policy    import GNNActorCritic
from mbqc_rl.agent.ppo           import PPOTrainer
from mbqc_rl.utils.metrics       import save_history, summarise, plot_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN mixed-rate PPO training")
    p.add_argument("--rows",           type=int,   default=3)
    p.add_argument("--cols",           type=int,   default=3)
    p.add_argument("--defect-lo",      type=float, default=0.0)
    p.add_argument("--defect-hi",      type=float, default=0.3)
    p.add_argument("--use-angles",     action="store_true")
    p.add_argument("--clifford-fraction", type=float, default=0.5)
    p.add_argument("--topology",       type=str,   default="grid",
                   choices=["grid", "brickwork", "irregular"])
    p.add_argument("--reward-shaping", action="store_true",
                   help="Dense per-step reward (same total as terminal score). "
                        "Needed for variable-structure tasks like irregular graphs.")
    p.add_argument("--observe-original", action="store_true",
                   help="Observe the original (undegraded) adjacency every step. "
                        "Preserves global gflow structure; helps on irregular graphs.")
    p.add_argument("--total-timesteps",type=int,   default=750_000)
    p.add_argument("--save-dir",       type=str,   default="results/gnn")
    # GNN architecture
    p.add_argument("--hidden-dim",     type=int,   default=64,
                   help="Node embedding width (divisible by n-heads)")
    p.add_argument("--n-heads",        type=int,   default=4,
                   help="Attention heads per GATConv layer")
    p.add_argument("--n-layers",       type=int,   default=3,
                   help="Number of GATConv layers")
    p.add_argument("--virtual-node",   action="store_true",
                   help="Add a global virtual-node readout after each GAT layer "
                        "(anti-over-squashing; ladder rung 1 for irregular graphs).")
    p.add_argument("--weight-tied",    action="store_true",
                   help="Share one GATConv across all n-layers applications "
                        "(iterative processor; ladder rung 2).")
    p.add_argument("--pos-dim",        type=int,   default=0,
                   help="Random-walk structural encoding dim appended to node "
                        "features (ladder rung 3). 0 = off.")
    p.add_argument("--aux-layer",      action="store_true",
                   help="Auxiliary head predicting per-node gflow layers, trained "
                        "with a supervised loss alongside PPO (ladder rung 4, NAR "
                        "hints). Use with --observe-original.")
    p.add_argument("--aux-weight",     type=float, default=1.0,
                   help="Weight of the auxiliary gflow-layer loss (if --aux-layer).")
    # PPO hyperparameters (same defaults as train_mixed.py)
    p.add_argument("--n-steps",        type=int,   default=512)
    p.add_argument("--n-epochs",       type=int,   default=10)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--weight-decay",   type=float, default=0.0,
                   help="Adam weight decay (L2 regularisation) — targets the "
                        "train↔held-out generalisation gap on irregular graphs.")
    p.add_argument("--skip-lo",        type=float, default=0.45,
                   help="Irregular skip-edge prob lower bound (= upper ⇒ fixed). "
                        "Set lo<hi to randomise irregularity per episode.")
    p.add_argument("--skip-hi",        type=float, default=0.45,
                   help="Irregular skip-edge prob upper bound.")
    p.add_argument("--gamma",          type=float, default=0.99)
    p.add_argument("--gae-lambda",     type=float, default=0.95)
    p.add_argument("--clip-range",     type=float, default=0.2)
    p.add_argument("--vf-coef",        type=float, default=0.5)
    p.add_argument("--ent-coef",       type=float, default=0.01)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--log-interval",   type=int,   default=10)
    p.add_argument("--init-from",      type=str,   default=None,
                   help="Path to a GNN checkpoint to initialise weights from "
                        "(transfer learning / fine-tuning). Works across grid "
                        "sizes because GNN weights are size-independent.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    n = args.rows * args.cols

    env = MBQCEnv(
        rows=args.rows, cols=args.cols,
        defect_rate=(args.defect_lo, args.defect_hi),
        use_angles=args.use_angles,
        clifford_fraction=args.clifford_fraction,
        topology=args.topology,
        reward_shaping=args.reward_shaping,
        observe_original_graph=args.observe_original,
        irregular_skip_prob=(args.skip_lo, args.skip_hi),
        seed=args.seed,
    )

    policy = GNNActorCritic(
        n          = n,
        hidden_dim = args.hidden_dim,
        n_heads    = args.n_heads,
        n_layers   = args.n_layers,
        use_angles = args.use_angles,
        virtual_node = args.virtual_node,
        weight_tied  = args.weight_tied,
        pos_dim      = args.pos_dim,
        aux_layer_head = args.aux_layer,
    )

    # Transfer learning: warm-start from another GNN checkpoint. The weights are
    # size-independent, so a 4×4-trained policy can initialise 5×5 fine-tuning.
    if args.init_from:
        src = torch.load(args.init_from, map_location="cpu", weights_only=False)
        policy.load_state_dict(src["policy_state_dict"])
        print(f"  Initialised weights from {args.init_from} (fine-tuning)")

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5,
                                 weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in policy.parameters())

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
        aux_weight = args.aux_weight if args.aux_layer else 0.0,
    )

    print(f"\n{'═'*60}")
    print(f"  GNN TRAINING: {args.rows}×{args.cols} grid  (n={n} qubits)")
    print(f"  Architecture: {args.n_layers}×GATConv  "
          f"hidden={args.hidden_dim}  heads={args.n_heads}")
    print(f"  Parameters: {n_params:,}  "
          f"(cf. MLP {n*n+n}→256→{n} ≈ {(n*n+n)*256 + 256*256 + 256*n:,})")
    print(f"  defect_rate ~ U[{args.defect_lo}, {args.defect_hi}]  "
          f"use_angles={args.use_angles}")
    print(f"  total timesteps: {args.total_timesteps:,}")
    print(f"{'═'*60}")

    history = trainer.train(
        total_timesteps = args.total_timesteps,
        save_dir        = os.path.join(args.save_dir, "checkpoints"),
        log_interval    = args.log_interval,
        save_interval   = 50,
    )

    # Save final checkpoint — includes model_type so loaders can dispatch
    ckpt_path = os.path.join(args.save_dir, "final.pt")
    torch.save({
        "policy_state_dict":    policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dict(
            model_type        = "gnn",
            rows              = args.rows,
            cols              = args.cols,
            n                 = n,
            obs_dim           = policy.obs_dim,
            n_actions         = n,
            hidden_dim        = args.hidden_dim,
            n_heads           = args.n_heads,
            n_layers          = args.n_layers,
            virtual_node      = args.virtual_node,
            weight_tied       = args.weight_tied,
            pos_dim           = args.pos_dim,
            aux_layer_head    = args.aux_layer,
            use_angles        = args.use_angles,
            clifford_fraction = args.clifford_fraction,
            topology          = args.topology,
            reward_shaping    = args.reward_shaping,
            observe_original  = args.observe_original,
            defect_lo         = args.defect_lo,
            defect_hi         = args.defect_hi,
        ),
    }, ckpt_path)
    print(f"\n  Final checkpoint → {ckpt_path}")

    save_history(history, os.path.join(args.save_dir, "history.json"))
    summarise(history)
    plot_training(
        history,
        save_to = os.path.join(args.save_dir, "training_curve.png"),
        title   = (f"GNN Training — {args.rows}×{args.cols} "
                   f"hidden={args.hidden_dim} heads={args.n_heads}"
                   + (" + angles" if args.use_angles else "")),
        show    = False,
    )


if __name__ == "__main__":
    main()
