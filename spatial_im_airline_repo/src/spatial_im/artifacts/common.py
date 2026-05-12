from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


REGIMES = ("spread", "dynamic", "spatial")


@dataclass(frozen=True)
class MainRegimeResult:
    regime: str
    rl_seeds: str
    rl_objective: float
    rl_mass: float
    rl_coverage: float
    rl_cost: float
    learned_env_objective: float
    best_baseline_name: str
    baseline_seeds: str
    baseline_objective: float
    baseline_mass: float
    baseline_coverage: float
    baseline_cost: float
    delta_objective: float
    submodularity_violation_rate: float


@dataclass(frozen=True)
class TransferRegimeResult:
    regime: str
    zero_shot_seeds: str
    warm_start_seeds: str
    scratch_seeds: str
    baseline_seeds: str
    zero_shot_raw_objective: float
    warm_start_raw_objective: float
    scratch_raw_objective: float
    baseline_raw_objective: float
    curve_transfer_ratio: float
    adaptation_efficiency_transfer: float
    adaptation_efficiency_scratch: float
    raw_transfer_ratio_warmstart_vs_scratch: float
    raw_transfer_ratio_zero_shot_vs_scratch: float
    raw_objective_gain_warmstart_vs_scratch: float
    raw_objective_gain_zero_shot_vs_scratch: float


@dataclass(frozen=True)
class PaperArtifactPaths:
    root: Path

    @property
    def figures_dir(self) -> Path:
        return self.root / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.root / "tables"

    @property
    def main_objective_figure(self) -> Path:
        return self.figures_dir / "fig_main_objective_comparison.png"

    @property
    def decomposition_figure(self) -> Path:
        return self.figures_dir / "fig_metric_decomposition.png"

    @property
    def transfer_figure(self) -> Path:
        return self.figures_dir / "fig_transfer_objectives.png"

    @property
    def main_results_md(self) -> Path:
        return self.tables_dir / "table_main_results.md"

    @property
    def main_results_tex(self) -> Path:
        return self.tables_dir / "table_main_results.tex"

    @property
    def transfer_results_md(self) -> Path:
        return self.tables_dir / "table_transfer_results.md"

    @property
    def transfer_results_tex(self) -> Path:
        return self.tables_dir / "table_transfer_results.tex"

    @property
    def manifest_json(self) -> Path:
        return self.root / "paper_minimal_manifest.json"


def _seed_str(seed_list: list[str]) -> str:
    return ", ".join(seed_list)


def load_main_results(path: str | Path) -> Dict[str, MainRegimeResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results: Dict[str, MainRegimeResult] = {}
    for regime in REGIMES:
        entry = data["regimes"][regime]
        rl = entry["rl"]
        rl_learned = entry.get("rl_learned_env", {})
        baseline = entry["best_baseline"]
        results[regime] = MainRegimeResult(
            regime=regime,
            rl_seeds=_seed_str(rl["seeds_iata"]),
            rl_objective=float(rl["objective"]),
            rl_mass=float(rl["final_activated_mass"]),
            rl_coverage=float(rl["expected_coverage"]),
            rl_cost=float(rl["intervention_cost"]),
            learned_env_objective=float(rl_learned.get("objective", rl["objective"])),
            best_baseline_name=str(entry["best_baseline_name"]),
            baseline_seeds=_seed_str(baseline["seeds_iata"]),
            baseline_objective=float(baseline["objective"]),
            baseline_mass=float(baseline["final_activated_mass"]),
            baseline_coverage=float(baseline["expected_coverage"]),
            baseline_cost=float(baseline["intervention_cost"]),
            delta_objective=float(rl["objective"]) - float(baseline["objective"]),
            submodularity_violation_rate=float(entry.get("submodularity_violation_rate", 0.0)),
        )
    return results


def load_transfer_results(path: str | Path) -> Dict[str, TransferRegimeResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results: Dict[str, TransferRegimeResult] = {}
    for regime in REGIMES:
        entry = data["regimes"][regime]
        metrics = entry["transfer_metrics"]
        baseline = entry["best_baseline"]
        results[regime] = TransferRegimeResult(
            regime=regime,
            zero_shot_seeds=_seed_str(entry["zero_shot"]["seeds_iata"]),
            warm_start_seeds=_seed_str(entry["transfer_adapt"]["seeds_iata"]),
            scratch_seeds=_seed_str(entry["scratch_adapt"]["seeds_iata"]),
            baseline_seeds=_seed_str(baseline["seeds_iata"]),
            zero_shot_raw_objective=float(entry["zero_shot"]["raw_objective"]),
            warm_start_raw_objective=float(entry["transfer_adapt"]["raw_objective"]),
            scratch_raw_objective=float(entry["scratch_adapt"]["raw_objective"]),
            baseline_raw_objective=float(baseline["objective"]),
            curve_transfer_ratio=float(metrics["curve_transfer_ratio"]),
            adaptation_efficiency_transfer=float(metrics["adaptation_efficiency_transfer"]),
            adaptation_efficiency_scratch=float(metrics["adaptation_efficiency_scratch"]),
            raw_transfer_ratio_warmstart_vs_scratch=float(metrics["raw_transfer_ratio_warmstart_vs_scratch"]),
            raw_transfer_ratio_zero_shot_vs_scratch=float(metrics["raw_transfer_ratio_zero_shot_vs_scratch"]),
            raw_objective_gain_warmstart_vs_scratch=float(metrics["raw_objective_gain_warmstart_vs_scratch"]),
            raw_objective_gain_zero_shot_vs_scratch=float(metrics["raw_objective_gain_zero_shot_vs_scratch"]),
        )
    return results
