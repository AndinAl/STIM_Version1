# Spatial Influence Maximization on Airline Networks (first prototype)

This repository is a **working first prototype** for influence maximization on a spatial airline network with **synthetic temporal edge weights**.

It follows the formulation where:
- the **environment** is a **GNN diffusion model**,
- the **policy** is a sequential RL agent,
- the same graph can be evaluated under **classical**, **dynamic**, and **spatial** regimes,
- the repo includes **baselines**, **evaluation metrics**, and **empirical submodularity diagnostics**.

## What this prototype does

1. Loads an airline graph from airport and route CSV files.
2. Computes spatial features such as coordinates and great-circle distances.
3. Generates **synthetic temporal edge weights** from the static airline topology.
4. Simulates progressive activation cascades to create training data.
5. Trains a **GNN diffusor** to predict activation probabilities over time.
6. Trains a **DQN-style RL policy** to sequentially pick seed nodes.
7. Runs baseline methods:
   - classical spread baselines,
   - dynamic centrality / myopic baselines,
   - spatial ranking baselines.
8. Reports evaluation metrics:
   - final activated mass,
   - geographic coverage,
   - intervention cost,
   - transfer ratio,
   - adaptation efficiency,
   - empirical submodularity-violation rate.

## Design choices

### Environment
The environment is a **learned one-step progressive GNN diffusor**:
- input: current activation probabilities, node features, edge weights, distances,
- output: next-step activation probabilities,
- rollout over a sequence of snapshots gives the final activation profile.

### Policy
The policy is a DQN-style scorer over candidate nodes using:
- current activation state,
- current snapshot summary,
- node centrality / spatial features,
- diffusor-predicted activation status.

This keeps the implementation simple and transparent for the first try.

### First-try airline setting
For the airline case, the repo assumes:
- fixed airport coordinates,
- fixed route topology,
- **time-varying synthetic route weights** generated from base traffic + seasonal factors + noise.

## Repository structure

```text
spatial_im_airline_repo/
├── README.md
├── requirements.txt
├── configs/
│   └── airline_synth.yaml
├── data/
│   └── sample_airline/
│       ├── airports.csv
│       └── routes.csv
├── scripts/
│   ├── quickstart.py
│   ├── train_diffusor.py
│   ├── train_rl.py
│   ├── run_baselines.py
│   └── evaluate.py
└── src/
    └── spatial_im/
        ├── data/
        │   ├── airline.py
        │   ├── graph_build.py
        │   └── temporal.py
        ├── diffusion/
        │   ├── simulator.py
        │   └── gnn_diffusor.py
        ├── env/
        │   └── airline_env.py
        ├── policy/
        │   ├── features.py
        │   └── dqn_agent.py
        ├── baselines/
        │   ├── classical.py
        │   ├── dynamic.py
        │   └── spatial.py
        ├── evaluation/
        │   ├── metrics.py
        │   ├── submodularity.py
        │   └── runner.py
        └── utils/
            ├── io.py
            └── seeds.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```bash
python scripts/quickstart.py --config configs/airline_synth.yaml
```

That script will:
1. build the graph,
2. generate temporal weights,
3. train a diffusor for a few epochs,
4. train an RL policy for a few episodes,
5. run baselines,
6. print a comparison table.

## Input format

### `airports.csv`
Required columns:
- `airport_id`
- `name`
- `iata`
- `lat`
- `lon`

### `routes.csv`
Required columns:
- `source_airport_id`
- `target_airport_id`

Extra columns are ignored.

## Regimes

The same code supports:

### 1. Spread-only regime
- one frozen snapshot,
- reward = activated mass only,
- used to check whether RL stays competitive with classical IM baselines.

### 2. Dynamic regime
- an episode is a sequence of snapshots,
- weights change over time,
- used to compare RL with weighted-degree and other myopic temporal baselines.

### 3. Spatial regime
- one frozen snapshot with spatial utility active,
- reward = activated mass + coverage - cost,
- used to compare RL with strength, betweenness, bridge, and cost-aware rankings.

## Important notes

- This is a **research prototype**, not a benchmark-optimized production system.
- The diffusor is intentionally lightweight and implemented only with **PyTorch**, not PyG, to keep the first try easy to run.
- The classical greedy baseline is **Monte Carlo greedy on the same graph/snapshot**.
- The dynamic baselines use the **same time windows** that the RL environment sees.
- The spatial baselines use **the same coordinates, distances, and node costs**.

## Suggested next steps

1. Replace the sample airline CSV files with OpenFlights or a cleaned airline dataset.
2. Replace synthetic temporal weights with real traffic/passenger time series if available.
3. Increase the number of Monte Carlo rollouts for more stable evaluation.
4. Add transfer experiments across time windows or route subgraphs.
5. Add a stricter monotone diffusion architecture if you want stronger progressive guarantees.
