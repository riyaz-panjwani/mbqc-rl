"""
Training metrics: saving, loading, summarising, and plotting.

Functions
---------
save_history(history, path)      Save a list-of-dicts training log as JSON.
load_history(path)               Load it back.
rolling_mean(values, window)     Smooth a sequence with a rolling window.
summarise(history)               Print a formatted training summary table.
plot_training(history, save_to)  Plot reward and loss curves with matplotlib.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_history(history: list[dict], path: str | Path) -> None:
    """Serialise a training history list to JSON (handles numpy floats)."""
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(history, f, indent=2, cls=_NumpyEncoder)


def load_history(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def rolling_mean(values: list[float], window: int = 20) -> np.ndarray:
    """
    Compute a rolling mean over `window` steps.
    Returns an array of the same length; early values use the available data.
    """
    arr = np.asarray(values, dtype=np.float64)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = arr[start : i + 1].mean()
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(history: list[dict], window: int = 50) -> None:
    """Print a table of key training statistics."""
    if not history:
        print("No history to summarise.")
        return

    rewards  = [h["mean_ep_reward"] for h in history]
    pl       = [h.get("policy_loss", 0.0) for h in history]
    vl       = [h.get("value_loss",  0.0) for h in history]
    ent      = [h.get("entropy",     0.0) for h in history]
    total_ts = history[-1].get("timestep", len(history))

    # Rolling mean over last `window` updates
    last_n   = rewards[-window:]

    print(f"\n{'─'*55}")
    print(f"  Training summary  ({len(history)} updates, {total_ts:,} timesteps)")
    print(f"{'─'*55}")
    print(f"  {'Metric':<25}  {'First 10':>10}  {'Last 50':>10}")
    print(f"  {'──────':<25}  {'────────':>10}  {'───────':>10}")

    def _fmt(seq):
        return f"{float(np.mean(seq)):.4f}" if seq else "   n/a"

    print(f"  {'Mean episode reward':<25}  {_fmt(rewards[:10]):>10}  {_fmt(last_n):>10}")
    print(f"  {'Policy loss':<25}  {_fmt(pl[:10]):>10}  {_fmt(pl[-window:]):>10}")
    print(f"  {'Value loss':<25}  {_fmt(vl[:10]):>10}  {_fmt(vl[-window:]):>10}")
    print(f"  {'Entropy':<25}  {_fmt(ent[:10]):>10}  {_fmt(ent[-window:]):>10}")
    print(f"{'─'*55}")
    print(f"  Peak reward  : {max(rewards):.4f}  (update {rewards.index(max(rewards)) + 1})")
    print(f"  Final reward : {rewards[-1]:.4f}")
    print(f"{'─'*55}\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training(
    history: list[dict],
    save_to: str | Path | None = None,
    title: str = "PPO Training",
    show: bool = True,
) -> None:
    """
    Plot episode reward and losses over training updates.

    Args:
        history:  Output of PPOTrainer.train().
        save_to:  If given, save the figure to this path (PNG/PDF).
        title:    Figure suptitle.
        show:     Call plt.show() if True.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not found — install with: pip install matplotlib")
        return

    updates  = [h["update"]          for h in history]
    rewards  = [h["mean_ep_reward"]  for h in history]
    pl       = [h.get("policy_loss", 0.0) for h in history]
    vl       = [h.get("value_loss",  0.0) for h in history]
    ent      = [h.get("entropy",     0.0) for h in history]
    smooth_r = rolling_mean(rewards, window=20)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Reward ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(updates, rewards,   alpha=0.25, color="steelblue", label="raw")
    ax0.plot(updates, smooth_r,  color="steelblue",             label="rolling mean (20)")
    ax0.axhline(1.0, ls="--", color="green",  alpha=0.6, label="perfect (1.0)")
    ax0.set_xlabel("Update")
    ax0.set_ylabel("Mean episode reward")
    ax0.set_title("Episode Reward")
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # ── Policy loss ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(updates, pl, color="tomato", lw=0.8)
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Loss")
    ax1.set_title("Policy Loss (L_clip)")
    ax1.grid(True, alpha=0.3)

    # ── Value loss ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(updates, vl,  color="orange", lw=0.8, label="value loss")
    ax2.plot(updates, ent, color="purple", lw=0.8, label="entropy")
    ax2.set_xlabel("Update")
    ax2.set_ylabel("Loss / Entropy")
    ax2.set_title("Value Loss & Entropy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"Figure saved → {save_to}")

    if show:
        plt.show()
    else:
        plt.close(fig)
