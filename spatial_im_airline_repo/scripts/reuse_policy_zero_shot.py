from __future__ import annotations

import argparse
import json

from spatial_im.training import reuse_policy_zero_shot
from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--regime', default=None, choices=['spread', 'dynamic', 'spatial'])
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--start-snapshot', type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['seed'])
    regime = args.regime or cfg['evaluation']['regime']
    budget = int(args.budget or cfg['rl']['budget'])
    result = reuse_policy_zero_shot(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        regime=regime,
        budget=budget,
        start_snapshot=args.start_snapshot,
    )
    print(
        json.dumps(
            {
                'regime': regime,
                'budget': budget,
                'start_snapshot': int(args.start_snapshot),
                'selected_idx': result.selected,
                'selected_iata': result.selected_iata,
                'learned_env_objective': result.learned_env_objective,
                'raw_objective': result.raw_objective,
                'raw_final_activated_mass': result.raw_final_activated_mass,
                'raw_expected_coverage': result.raw_expected_coverage,
                'raw_intervention_cost': result.raw_intervention_cost,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
