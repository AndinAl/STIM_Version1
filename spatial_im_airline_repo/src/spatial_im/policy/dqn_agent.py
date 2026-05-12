from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Deque, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.spread_head = nn.Linear(hidden_dim, 1)
        self.cover_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)

    def forward_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        q_spread = self.spread_head(h).squeeze(-1)
        q_cover = self.cover_head(h).squeeze(-1)
        q_cost = self.cost_head(h).squeeze(-1)
        return q_spread, q_cover, q_cost

    def forward(self, x: torch.Tensor, beta_coverage: float = 0.0, lambda_cost: float = 0.0) -> torch.Tensor:
        q_spread, q_cover, q_cost = self.forward_components(x)
        return q_spread + beta_coverage * q_cover - lambda_cost * q_cost


@dataclass
class Transition:
    state_feat: np.ndarray
    action: int
    reward: float
    next_state_feat: np.ndarray
    done: bool
    legal_mask: np.ndarray
    next_legal_mask: np.ndarray
    action_prior: np.ndarray | None = None
    next_action_prior: np.ndarray | None = None


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state_feat: np.ndarray,
        action: int,
        reward: float,
        next_state_feat: np.ndarray,
        done: bool,
        legal_mask: np.ndarray,
        next_legal_mask: np.ndarray,
        action_prior: np.ndarray | None = None,
        next_action_prior: np.ndarray | None = None,
    ):
        self.buffer.append(
            Transition(
                state_feat=state_feat,
                action=action,
                reward=reward,
                next_state_feat=next_state_feat,
                done=done,
                legal_mask=legal_mask,
                next_legal_mask=next_legal_mask,
                action_prior=action_prior,
                next_action_prior=next_action_prior,
            )
        )

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        feature_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        hidden_dim: int = 64,
        beta_coverage: float = 0.0,
        lambda_cost: float = 0.0,
        regime: str | None = None,
        action_prior_weight: float | None = None,
        device='cpu',
    ):
        self.q = QScorer(feature_dim, hidden_dim=hidden_dim).to(device)
        self.target_q = QScorer(feature_dim, hidden_dim=hidden_dim).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.beta_coverage = beta_coverage
        self.lambda_cost = lambda_cost
        self.regime = regime
        if action_prior_weight is None:
            if regime == 'spread':
                action_prior_weight = 1.0
            elif regime == 'dynamic':
                action_prior_weight = 0.5
            else:
                action_prior_weight = 0.0
        self.action_prior_weight = float(action_prior_weight)
        self.device = device

    @staticmethod
    def _legal_candidates(legal_mask: np.ndarray) -> np.ndarray:
        legal_idx = np.where(legal_mask > 0)[0]
        if len(legal_idx) == 0:
            raise RuntimeError('No legal actions left.')
        return legal_idx

    def _apply_action_prior(
        self,
        q_legal: torch.Tensor,
        legal_idx: np.ndarray,
        action_prior: np.ndarray | None,
    ) -> torch.Tensor:
        if action_prior is None or self.action_prior_weight == 0.0:
            return q_legal
        prior = torch.as_tensor(action_prior[legal_idx], dtype=torch.float32, device=self.device)
        return q_legal + self.action_prior_weight * prior

    def score_legal_candidates(
        self,
        node_features: np.ndarray,
        legal_mask: np.ndarray,
        action_prior: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        legal_idx = self._legal_candidates(legal_mask)
        with torch.no_grad():
            x = torch.as_tensor(node_features[legal_idx], dtype=torch.float32, device=self.device)
            q_legal = self.q(x, beta_coverage=self.beta_coverage, lambda_cost=self.lambda_cost)
            q_legal = self._apply_action_prior(q_legal, legal_idx, action_prior).cpu().numpy()
        return legal_idx, q_legal

    def act(
        self,
        node_features: np.ndarray,
        legal_mask: np.ndarray,
        epsilon: float,
        action_prior: np.ndarray | None = None,
    ) -> int:
        legal_idx = self._legal_candidates(legal_mask)
        if random.random() < epsilon:
            return int(random.choice(legal_idx.tolist()))
        legal_idx, q_legal = self.score_legal_candidates(node_features, legal_mask, action_prior=action_prior)
        return int(legal_idx[int(np.argmax(q_legal))])

    def greedy_action(
        self,
        node_features: np.ndarray,
        legal_mask: np.ndarray,
        action_prior: np.ndarray | None = None,
    ) -> int:
        legal_idx, q_legal = self.score_legal_candidates(node_features, legal_mask, action_prior=action_prior)
        return int(legal_idx[int(np.argmax(q_legal))])

    def update(self, batch: List[Transition]) -> float:
        if not batch:
            return 0.0
        losses = []
        for tr in batch:
            x = torch.as_tensor(tr.state_feat, dtype=torch.float32, device=self.device)
            q_vals = self.q(x, beta_coverage=self.beta_coverage, lambda_cost=self.lambda_cost)
            q_vals = self._apply_action_prior(q_vals, np.arange(x.size(0)), tr.action_prior)
            q_sa = q_vals[tr.action]

            with torch.no_grad():
                nx = torch.as_tensor(tr.next_state_feat, dtype=torch.float32, device=self.device)
                nq = self.target_q(nx, beta_coverage=self.beta_coverage, lambda_cost=self.lambda_cost)
                nq = self._apply_action_prior(nq, np.arange(nx.size(0)), tr.next_action_prior)
                next_mask = torch.as_tensor(tr.next_legal_mask, dtype=torch.bool, device=self.device)
                if next_mask.any() and not tr.done:
                    nq = nq.masked_fill(~next_mask, -1e9)
                    target = torch.tensor(tr.reward, dtype=torch.float32, device=self.device) + self.gamma * torch.max(nq)
                else:
                    target = torch.tensor(tr.reward, dtype=torch.float32, device=self.device)
            loss = F.mse_loss(q_sa, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def sync_target(self):
        self.target_q.load_state_dict(self.q.state_dict())

    def state_dict(self) -> dict:
        return {
            'q_state_dict': self.q.state_dict(),
            'target_q_state_dict': self.target_q.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'gamma': self.gamma,
            'beta_coverage': self.beta_coverage,
            'lambda_cost': self.lambda_cost,
            'regime': self.regime,
            'action_prior_weight': self.action_prior_weight,
            'device': self.device,
        }

    def load_state_dict(self, state: dict, strict: bool = True):
        self.q.load_state_dict(state['q_state_dict'], strict=strict)
        target_state = state.get('target_q_state_dict', state['q_state_dict'])
        self.target_q.load_state_dict(target_state, strict=strict)
        optimizer_state = state.get('optimizer_state_dict')
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.regime = state.get('regime', self.regime)
        self.action_prior_weight = state.get('action_prior_weight', self.action_prior_weight)
