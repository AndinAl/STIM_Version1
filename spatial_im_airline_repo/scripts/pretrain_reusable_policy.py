from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from spatial_im.training import pretrain_reusable_policy
from spatial_im.utils.io import load_yaml
from spatial_im.utils.seeds import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', default='artifacts/pretrained_reusable_policy.pt')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['seed'])
    artifacts = pretrain_reusable_policy(cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts.checkpoint()
    checkpoint['config'] = cfg
    torch.save(checkpoint, out)

    returns = np.asarray(artifacts.history['episode_return'], dtype=np.float32)
    diffusor_losses = np.asarray(artifacts.history['diffusor_loss'], dtype=np.float32)
    print('Saved reusable pretrained policy to', out)
    print('Source graphs:', ', '.join(artifacts.source_graphs))
    print('Episodes:', int(returns.size))
    if returns.size:
        print('Last 10 episode mean return:', float(returns[-10:].mean()))
    if diffusor_losses.size:
        print('Last 10 diffusor mean loss:', float(diffusor_losses[-10:].mean()))


if __name__ == '__main__':
    main()
