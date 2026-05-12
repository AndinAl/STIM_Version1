from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from spatial_im.training import reuse_across_future_snapshots
from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--regime', default=None, choices=['spread', 'dynamic', 'spatial'])
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--train-snapshot-count', type=int, required=True)
    parser.add_argument('--test-start-snapshot', type=int, required=True)
    parser.add_argument('--pretrained-out', default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['seed'])
    regime = args.regime or cfg['evaluation']['regime']
    budget = int(args.budget or cfg['rl']['budget'])
    pretrained, result = reuse_across_future_snapshots(
        cfg=cfg,
        regime=regime,
        budget=budget,
        train_snapshot_count=args.train_snapshot_count,
        test_start_snapshot=args.test_start_snapshot,
    )

    if args.pretrained_out:
        out = Path(args.pretrained_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = pretrained.checkpoint()
        checkpoint['config'] = cfg
        torch.save(checkpoint, out)

    print(
        json.dumps(
            {
                'regime': regime,
                'budget': budget,
                'train_snapshot_count': int(args.train_snapshot_count),
                'test_start_snapshot': int(args.test_start_snapshot),
                'selected_idx': result.selected,
                'selected_iata': result.selected_iata,
                'learned_env_objective': result.learned_env_objective,
                'raw_objective': result.raw_objective,
                'raw_final_activated_mass': result.raw_final_activated_mass,
                'raw_expected_coverage': result.raw_expected_coverage,
                'raw_intervention_cost': result.raw_intervention_cost,
                'pretrained_out': args.pretrained_out,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
