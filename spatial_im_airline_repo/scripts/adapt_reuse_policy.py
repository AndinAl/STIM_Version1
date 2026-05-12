from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from spatial_im.training import adapt_and_reuse_policy
from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--regime', default=None, choices=['spread', 'dynamic', 'spatial'])
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--start-snapshot', type=int, default=0)
    parser.add_argument('--adapt-iters', type=int, default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['seed'])
    regime = args.regime or cfg['evaluation']['regime']
    budget = int(args.budget or cfg['rl']['budget'])
    artifacts = adapt_and_reuse_policy(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        regime=regime,
        budget=budget,
        start_snapshot=args.start_snapshot,
        n_adapt=args.adapt_iters,
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts.checkpoint(cfg), out)

    returns = np.asarray(artifacts.history['episode_return'], dtype=np.float32)
    diffusor_losses = np.asarray(artifacts.history['diffusor_loss'], dtype=np.float32)
    print(
        json.dumps(
            {
                'regime': regime,
                'budget': budget,
                'start_snapshot': int(args.start_snapshot),
                'adapt_iterations': int(args.adapt_iters or cfg.get('adaptation', {}).get('iterations', 5)),
                'selected_idx': artifacts.result.selected,
                'selected_iata': artifacts.result.selected_iata,
                'learned_env_objective': artifacts.result.learned_env_objective,
                'raw_objective': artifacts.result.raw_objective,
                'raw_final_activated_mass': artifacts.result.raw_final_activated_mass,
                'raw_expected_coverage': artifacts.result.raw_expected_coverage,
                'raw_intervention_cost': artifacts.result.raw_intervention_cost,
                'adapt_episode_count': int(np.isfinite(returns).sum()),
                'last10_adapt_return_mean': float(np.nanmean(returns[-10:])) if returns.size else None,
                'last10_diffusor_loss_mean': float(np.nanmean(diffusor_losses[-10:])) if diffusor_losses.size else None,
                'saved_checkpoint': args.out,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
