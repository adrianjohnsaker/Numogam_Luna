# -*- coding: utf-8 -*-
"""
rhythmic_process_sequencer.py
──────────────────────────────────────────────────────────────────────────────
Rhythmic Process Sequencer (RPS)

Process-philosophy inspired timing & learning modulation:
- Tracks sliding novelty (prediction error + optional online micro-clustering)
- Slows internal clock when novelty spikes ("dwell"), speeds when novelty wanes
- Modulates learning rate with novelty
- Emits headers for Amelia's pipeline (temperature, exploration, fold nudges)

Dependencies: numpy
"""

from __future__ import annotations
import time, json, math, random
from typing import Dict, Any, Optional, Tuple
import numpy as np


def _softclip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class OnlineCentroids:
    """
    Tiny online k-centroid sketch (no labels, no batch).
    Keeps K centroids; updates nearest with small step (lr_cluster).
    Used to measure novelty as distance from the closest known mode.
    """
    def __init__(self, dim: int, K: int = 6, seed_scale: float = 1e-3):
        self.K = K
        self.dim = dim
        self.ready = False
        self.C = np.zeros((K, dim), dtype=np.float32)
        self.seed_scale = seed_scale

    def _init_if_needed(self, x: np.ndarray):
        if not self.ready:
            # seed small random jitter around the first vector
            self.C = x[None, :] + np.random.randn(self.K, self.dim).astype(np.float32) * self.seed_scale
            self.ready = True

    def nearest(self, x: np.ndarray) -> Tuple[int, float]:
        self._init_if_needed(x)
        d2 = np.sum((self.C - x[None, :])**2, axis=1)
        idx = int(np.argmin(d2))
        return idx, float(np.sqrt(d2[idx]) + 1e-9)

    def update(self, x: np.ndarray, lr_cluster: float = 0.04) -> float:
        self._init_if_needed(x)
        idx, dist = self.nearest(x)
        self.C[idx] = (1.0 - lr_cluster) * self.C[idx] + lr_cluster * x
        return dist


class RhythmicProcessSequencer:
    """
    Maintains:
      • novelty (EMA of prediction error & centroid distance)
      • internal tempo (Hz) that slows on novelty spikes
      • learning rates that scale with novelty

    Emits:
      • sleep seconds for your loop
      • headers for pipeline (policy + RIM)
    """
    def __init__(
        self,
        dim: int,
        base_hz: float = 1.0,
        min_hz: float = 0.20,
        max_hz: float = 2.50,
        lr_base: float = 0.05,
        lr_min: float = 0.01,
        lr_max: float = 0.25,
        ema_alpha: float = 0.08,
        novelty_alpha: float = 0.12,
        cluster_K: int = 6,
        cluster_lr: float = 0.04,
        seed: Optional[int] = None,
    ):
        self.dim = dim
        self.base_hz = base_hz
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.lr_base = lr_base
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.ema_alpha = ema_alpha
        self.novelty_alpha = novelty_alpha
        self.cluster_lr = cluster_lr

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # predictive state (simple EMA predictor)
        self.mu = np.zeros((dim,), dtype=np.float32)
        self.sqerr_ema = 1e-6  # keeps scale for normalized error

        # centroid sketch
        self.clust = OnlineCentroids(dim=dim, K=cluster_K)

        # working state
        self.novelty = 0.0
        self.last_step_ts = time.time()
        self.last_hz = base_hz
        self.last_lr = lr_base

    # ------------------------------ Core Update ------------------------------

    def update(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Update sequencer with a new observation vector x (shape [dim]).
        Returns dict: novelty, tempo_hz, sleep_s, lr, headers, debug
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        assert x.shape[0] == self.dim, "x must match sequencer dim"

        # Prediction error against EMA
        pred = self.mu
        err = x - pred
        sqerr = float(np.dot(err, err))
        # Normalize error by running scale (RMS)
        self.sqerr_ema = (1 - self.ema_alpha) * self.sqerr_ema + self.ema_alpha * sqerr
        rms = math.sqrt(self.sqerr_ema) + 1e-9
        pe_norm = _softclip(sqerr / (rms + 1e-9), 0.0, 50.0)  # bounded

        # Micro-cluster novelty (distance to nearest centroid)
        dist = self.clust.update(x, lr_cluster=self.cluster_lr)
        # Normalize by robust scale (use RMS as a proxy)
        dist_norm = _softclip(dist / (rms + 1e-9), 0.0, 50.0)

        # Combined novelty (smooth)
        obs_novelty = 0.6 * pe_norm + 0.4 * dist_norm
        # squash to [0,1] with logistic-ish compression
        obs_nov_squash = 1.0 - math.exp(-0.08 * obs_novelty)
        self.novelty = (1 - self.novelty_alpha) * self.novelty + self.novelty_alpha * obs_nov_squash
        nov = _softclip(self.novelty, 0.0, 1.0)

        # Tempo mapping: high novelty → slow down (dwell)
        # hz = base * mix(low_on_high_nov, high_on_low_nov)
        slow = self.base_hz * 0.45
        fast = self.base_hz * 1.35
        hz = (1 - nov) * fast + nov * slow
        hz = _softclip(hz, self.min_hz, self.max_hz)
        self.last_hz = hz

        # Learning rate mapping: higher novelty → learn faster (but bounded)
        lr = self.lr_base * (0.6 + 0.9 * nov)
        lr = _softclip(lr, self.lr_min, self.lr_max)
        self.last_lr = lr

        # Update EMA predictor with novelty-scaled learning
        self.mu = (1.0 - lr) * self.mu + lr * x

        # Sleep time for loop pacing
        sleep_s = 1.0 / hz

        headers = self._emit_headers(nov, hz, lr)

        # Bookkeeping
        self.last_step_ts = time.time()

        return {
            "novelty": round(nov, 4),
            "tempo_hz": round(hz, 4),
            "sleep_s": round(sleep_s, 4),
            "lr": round(lr, 4),
            "headers": headers,
            "debug": {
                "pe_norm": round(pe_norm, 4),
                "dist_norm": round(dist_norm, 4),
                "sqerr_ema": round(self.sqerr_ema, 6),
                "rms": round(rms, 6),
            }
        }

    # ------------------------------ Headers Out ------------------------------

    def _emit_headers(self, novelty: float, hz: float, lr: float) -> Dict[str, str]:
        """
        Translate rhythmic state into Amelia steering hints.
        - Higher novelty → lower temperature cap? (we dwell), but increase exploration_bias subtly
        - Faster tempo (low novelty) → encourage wider exploration
        - Fold nudge: at high novelty, reduce fold a bit to keep coherence in dwell
        """
        # temperature adapts mostly with speed; exploration with novelty
        temp = 0.7 + 0.25 * ((hz - self.min_hz) / (self.max_hz - self.min_hz + 1e-9))  # ~[0.7,0.95]
        explore = 0.95 + 0.4 * (novelty)  # [0.95, 1.35]
        fold_nudge = 1.0 - 0.18 * novelty  # reduce folding at high novelty for focus

        policy = {
            "resonance_nudge": round((0.5 - novelty) * 0.2, 3),  # dwell → +resonance
            "temperature": round(temp, 3)
        }

        rim = {
            "exploration_bias": round(explore, 3),
            "coherence_bonus": round(0.95 + 0.15 * (1.0 - novelty), 3),
            "symbolic_weight": round(0.98 + 0.1 * (1.0 - novelty), 3),
            "temporal_stability": round(0.95 + 0.25 * (1.0 - novelty), 3),
            "fold_nudge": round(fold_nudge, 3)
        }

        return {
            "X-Amelia-Policy": json.dumps(policy),
            "X-Amelia-RIM": json.dumps(rim),
            "X-Amelia-Rhythm": json.dumps({
                "novelty": round(novelty, 4),
                "tempo_hz": round(hz, 4),
                "lr": round(lr, 4)
            })
        }

    # ------------------------------ Helpers ------------------------------

    def next_sleep_seconds(self) -> float:
        """Convenience accessor for loop schedulers."""
        return 1.0 / (self.last_hz if self.last_hz > 0 else self.base_hz)
