#!/usr/bin/env python3
"""Readable reference implementation for the HFX active-set workload.

The module is intentionally standalone so that the theorem-aligned mechanism,
memory accounting, reference methods, and small smoke tests can be inspected or
run without compiling the authoritative C++ implementation.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


MASK64 = (1 << 64) - 1
GOLDEN64 = 0x9E3779B97F4A7C15


def splitmix64(x: int) -> int:
    x = (x + GOLDEN64) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return (x ^ (x >> 31)) & MASK64


def hash_pair(item: int, seed: int) -> tuple[int, int]:
    h1 = splitmix64((item ^ (seed * GOLDEN64)) & MASK64)
    h2 = splitmix64((h1 ^ (seed << 32) ^ 0xD1B54A32D192ED03) & MASK64)
    return h1, h2


def ctz64(x: int) -> int:
    if x == 0:
        return 64
    return ((x & -x).bit_length() - 1)


def trailing_ones32(x: int) -> int:
    """Number of consecutive one bits from the least-significant side."""
    x &= 0xFFFFFFFF
    first_zero_mask = (~x) & 0xFFFFFFFF
    if first_zero_mask == 0:
        return 32
    return ctz64(first_zero_mask)


def parse_number_list(text: str, cast):
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def epsilon_delta_to_rho(epsilon: float, delta: float) -> float:
    log_delta = math.log(1.0 / delta)
    return (math.sqrt(log_delta + epsilon) - math.sqrt(log_delta)) ** 2


def tree_height(T: int) -> int:
    return max(1, math.ceil(math.log2(T + 1)))


def default_nmax(T: int) -> int:
    if T < 10_000_000:
        return max(1_000, T // 10)
    if T == 10_000_000:
        return 1_000_000
    if T == 100_000_000:
        return 10_000_000
    if T == 1_000_000_000:
        return 100_000_000
    return max(1_000, T // 10)


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


class GaussianBinaryTree:
    def __init__(self, T: int, sigma: float, rng: np.random.Generator):
        self.H = tree_height(T)
        self.U = np.zeros(self.H + 1, dtype=np.float64)
        self.P = np.zeros(self.H + 1, dtype=np.float64)
        self.sigma = float(sigma)
        self.rng = rng
        self.active_noisy_sum = 0.0

    def update(self, t_one_based: int, impulse: float) -> float:
        idx = ctz64(t_one_based)
        partial = float(impulse)
        if idx > 0:
            partial += float(np.sum(self.U[:idx]))
            self.active_noisy_sum -= float(np.sum(self.P[:idx]))
            self.U[:idx] = 0.0
            self.P[:idx] = 0.0
        self.U[idx] = partial
        self.P[idx] = partial + self.rng.normal(0.0, self.sigma)
        self.active_noisy_sum += float(self.P[idx])
        return self.active_noisy_sum


class HfxEstimator:
    def __init__(self, m: int, w: int, n_max: int):
        self.m = int(m)
        self.w = int(w)
        self.n_max = int(n_max)
        js = np.arange(1, w + 1, dtype=np.float64)
        probs = np.power(2.0, -js)
        probs[-1] = 2.0 ** (-(w - 1))
        self.log_terms = np.log1p(-2.0 * probs / float(m))
        self.max_phi = self.phi(float(n_max))

    def phi(self, n: float) -> float:
        # Exact parity expectation from Algorithm 2:
        # Phi(n) = m/2 * sum_j (1 - (1 - 2 p_j / m)^n).
        return 0.5 * self.m * float(np.sum(-np.expm1(n * self.log_terms)))

    def invert(self, z_observed: float) -> float:
        z = min(max(float(z_observed), 0.0), self.max_phi)
        lo = 0.0
        hi = float(self.n_max)
        for _ in range(48):
            mid = (lo + hi) * 0.5
            if self.phi(mid) < z:
                lo = mid
            else:
                hi = mid
        return (lo + hi) * 0.5


@dataclass
class MethodResult:
    method: str
    rmsd: float
    relative_error: float
    accepted_updates: int
    filtered_updates: int
    saturated_cells: int
    num_cells: int
    load_p50: Optional[float] = None
    load_p95: Optional[float] = None
    load_p99: Optional[float] = None
    load_max: Optional[float] = None
    throughput_mops: float = 0.0
    sigma_node: Optional[float] = None
    clipped_evals: int = 0
    eval_samples: int = 0


class HFXGaussian:
    method = "HFX"

    def __init__(
        self,
        *,
        T: int,
        n_max: int,
        memory_kb: int,
        epsilon: float,
        delta: float,
        q: int,
        w: int,
        seed_hash: int,
        seed_noise: int,
        noise_mode: str = "load_aware",
    ):
        self.T = int(T)
        self.n_max = int(n_max)
        self.memory_kb = int(memory_kb)
        self.q = int(q)
        self.R = (1 << q) - 1
        self.w = int(w)
        self.H = tree_height(T)
        self.rho = epsilon_delta_to_rho(epsilon, delta)
        self.m_tree_bits = 2 * (self.H + 1) * 64
        memory_bits = memory_kb * 1024 * 8
        available = max(0, memory_bits - self.m_tree_bits)
        self.num_cells = max(w, (available // (1 + q) // w) * w)
        self.m = max(1, self.num_cells // w)
        self.num_cells = self.m * w
        self.seed_hash = int(seed_hash)

        self.B = np.zeros(self.m, dtype=np.uint32)
        counter_dtype = np.uint8 if q <= 8 else np.uint16 if q <= 16 else np.uint32
        self.V = np.zeros(self.num_cells, dtype=counter_dtype)
        self.raw_load = np.zeros(self.num_cells, dtype=np.uint32)

        self.noise_mode = noise_mode
        if noise_mode in {"load_aware", "theorem"}:
            sigma_sq = 2.0 * (self.R + 1) * self.H / self.rho
        elif noise_mode in {"unit_diagnostic", "unit"}:
            sigma_sq = 2.0 * self.H / self.rho
        else:
            raise ValueError(f"Unknown HFX noise mode: {noise_mode}")
        self.sigma_node = math.sqrt(sigma_sq)
        self.tree = GaussianBinaryTree(T, self.sigma_node, np.random.default_rng(seed_noise))
        self.estimator = HfxEstimator(self.m, self.w, n_max)

        self.accepted = 0
        self.filtered = 0
        self.last_z = 0.0
        self.last_estimate = 0.0
        self.clipped_evals = 0
        self.eval_samples = 0

    def update(self, t_one_based: int, item: int, op: int) -> None:
        h1, h2 = hash_pair(item, self.seed_hash)
        row = h1 % self.m
        col = min(ctz64(h2), self.w - 1)
        coord = row * self.w + col
        self.raw_load[coord] += 1

        if int(self.V[coord]) >= self.R:
            self.filtered += 1
            impulse = 0.0
        else:
            self.V[coord] += 1
            bit = np.uint32(1 << col)
            old_is_one = (int(self.B[row]) & int(bit)) != 0
            self.B[row] ^= bit
            impulse = -1.0 if old_is_one else 1.0
            self.accepted += 1

        self.last_z = self.tree.update(t_one_based, impulse)

    def estimate(self) -> float:
        self.eval_samples += 1
        if self.last_z < 0.0 or self.last_z > self.estimator.max_phi:
            self.clipped_evals += 1
        self.last_estimate = self.estimator.invert(self.last_z)
        return self.last_estimate

    def finish(self, rmsd: float, final_true: int, elapsed: float) -> MethodResult:
        saturated = int(np.count_nonzero(self.V == self.R))
        rel = 0.0 if final_true == 0 else (self.last_estimate - final_true) / final_true
        nonzero_loads = self.raw_load[self.raw_load > 0]
        return MethodResult(
            method=self.method,
            rmsd=rmsd,
            relative_error=rel,
            accepted_updates=self.accepted,
            filtered_updates=self.filtered,
            saturated_cells=saturated,
            num_cells=self.num_cells,
            load_p50=percentile(nonzero_loads, 50),
            load_p95=percentile(nonzero_loads, 95),
            load_p99=percentile(nonzero_loads, 99),
            load_max=float(nonzero_loads.max()) if nonzero_loads.size else 0.0,
            throughput_mops=(self.T / elapsed / 1_000_000.0) if elapsed > 0 else 0.0,
            sigma_node=self.sigma_node,
            clipped_evals=self.clipped_evals,
            eval_samples=self.eval_samples,
        )


class ScalarGaussianTreeMethod:
    def __init__(self, *, T: int, n_max: int, epsilon: float, delta: float, sensitivity: float, seed_noise: int):
        rho = epsilon_delta_to_rho(epsilon, delta)
        H = tree_height(T)
        sigma = sensitivity * math.sqrt(H / (2.0 * rho))
        self.T = T
        self.n_max = n_max
        self.tree = GaussianBinaryTree(T, sigma, np.random.default_rng(seed_noise))
        self.last_z = 0.0
        self.last_estimate = 0.0
        self.current_t = 0

    def finish(self, method: str, rmsd: float, final_true: int, elapsed: float, num_cells: int) -> MethodResult:
        rel = 0.0 if final_true == 0 else (self.last_estimate - final_true) / final_true
        return MethodResult(
            method=method,
            rmsd=rmsd,
            relative_error=rel,
            accepted_updates=self.T,
            filtered_updates=0,
            saturated_cells=0,
            num_cells=num_cells,
            throughput_mops=(self.T / elapsed / 1_000_000.0) if elapsed > 0 else 0.0,
        )


class SplitSketchTurnstileMethod:
    """Turn insertion-only sketches into a turnstile baseline by subtraction.

    The add stream and delete stream each get half of the zCDP budget.  This is
    closer to the simple FM/LL/HLL turnstile baselines used in the old code than
    the previous exact per-coordinate counter simulation, and it avoids a large
    unaccounted state table for deletions.
    """

    method = "SplitSketch"

    def __init__(
        self,
        *,
        T: int,
        n_max: int,
        memory_kb: int,
        epsilon: float,
        delta: float,
        sensitivity: float,
        seed_hash: int,
        seed_noise: int,
    ):
        rho = epsilon_delta_to_rho(epsilon, delta)
        H = tree_height(T)
        sigma = sensitivity * math.sqrt(H / rho)
        self.T = int(T)
        self.n_max = int(n_max)
        self.memory_bits = int(memory_kb) * 1024 * 8
        self.seed_hash = int(seed_hash)
        self.tree_add = GaussianBinaryTree(T, sigma, np.random.default_rng(seed_noise))
        self.tree_del = GaussianBinaryTree(T, sigma, np.random.default_rng(seed_noise + 1_000_003))
        self.last_add_z = 0.0
        self.last_del_z = 0.0
        self.last_estimate = 0.0
        self.current_t = 0
        self.num_inserts = 0
        self.num_deletes = 0

    def update(self, t_one_based: int, item: int, op: int) -> None:
        self.current_t = t_one_based
        delta_add = 0.0
        delta_del = 0.0
        if op > 0:
            self.num_inserts += 1
            delta_add = self._insert_add(item)
        else:
            self.num_deletes += 1
            delta_del = self._insert_delete(item)
        self.last_add_z = self.tree_add.update(t_one_based, delta_add)
        self.last_del_z = self.tree_del.update(t_one_based, delta_del)

    def _insert_add(self, item: int) -> float:
        raise NotImplementedError

    def _insert_delete(self, item: int) -> float:
        raise NotImplementedError

    def _estimate_one(self, z: float, seen: int) -> float:
        raise NotImplementedError

    def estimate(self) -> float:
        add_est = self._estimate_one(self.last_add_z, self.num_inserts)
        del_est = self._estimate_one(self.last_del_z, self.num_deletes)
        estimate = add_est - del_est
        self.last_estimate = max(0.0, min(self.n_max, self.current_t, estimate))
        return self.last_estimate

    def result(self, rmsd: float, final_true: int, elapsed: float) -> MethodResult:
        rel = 0.0 if final_true == 0 else (self.last_estimate - final_true) / final_true
        return MethodResult(
            method=self.method,
            rmsd=rmsd,
            relative_error=rel,
            accepted_updates=self.T,
            filtered_updates=0,
            saturated_cells=0,
            num_cells=self.num_cells,
            throughput_mops=(self.T / elapsed / 1_000_000.0) if elapsed > 0 else 0.0,
        )


class FMTurnstile(SplitSketchTurnstileMethod):
    method = "FM-Turnstile"

    def __init__(self, *, T: int, n_max: int, memory_kb: int, epsilon: float, delta: float, seed_hash: int, seed_noise: int):
        super().__init__(
            T=T,
            n_max=n_max,
            memory_kb=memory_kb,
            epsilon=epsilon,
            delta=delta,
            sensitivity=64.0,
            seed_hash=seed_hash,
            seed_noise=seed_noise,
        )
        per_sketch_bits = max(32, self.memory_bits // 2)
        self.m = max(1, per_sketch_bits // 32)
        self.add_bitmap = np.zeros(self.m, dtype=np.uint32)
        self.del_bitmap = np.zeros(self.m, dtype=np.uint32)
        self.num_cells = 2 * self.m * 32

    def _insert_into(self, bitmap: np.ndarray, item: int) -> float:
        h1, h2 = hash_pair(item, self.seed_hash)
        row = h1 % self.m
        col = min(ctz64(h2), 31)
        old_row = int(bitmap[row])
        old_prefix = trailing_ones32(old_row)
        bitmap[row] = np.uint32(old_row | (1 << col))
        new_prefix = trailing_ones32(int(bitmap[row]))
        return float(new_prefix - old_prefix)

    def _insert_add(self, item: int) -> float:
        return self._insert_into(self.add_bitmap, item)

    def _insert_delete(self, item: int) -> float:
        return self._insert_into(self.del_bitmap, item)

    def _estimate_one(self, z: float, seen: int) -> float:
        if seen <= 0 or z <= 0.0:
            return 0.0
        estimate = self.m * (2.0 ** (z / self.m)) / 0.77351
        return max(0.0, min(float(max(self.T, self.n_max)), estimate))


class LLTurnstile(SplitSketchTurnstileMethod):
    method = "LL-Turnstile"

    def __init__(self, *, T: int, n_max: int, memory_kb: int, epsilon: float, delta: float, seed_hash: int, seed_noise: int):
        super().__init__(
            T=T,
            n_max=n_max,
            memory_kb=memory_kb,
            epsilon=epsilon,
            delta=delta,
            sensitivity=64.0,
            seed_hash=seed_hash,
            seed_noise=seed_noise,
        )
        per_sketch_bits = max(6, self.memory_bits // 2)
        self.m = max(1, per_sketch_bits // 6)
        self.max_rank = 64
        self.add_registers = np.zeros(self.m, dtype=np.uint8)
        self.del_registers = np.zeros(self.m, dtype=np.uint8)
        self.num_cells = 2 * self.m

    def _insert_into(self, registers: np.ndarray, item: int) -> float:
        h1, h2 = hash_pair(item, self.seed_hash)
        row = h1 % self.m
        rank = min(ctz64(h2) + 1, self.max_rank)
        old_reg = int(registers[row])
        if rank > old_reg:
            registers[row] = rank
            return float(rank - old_reg)
        return 0.0

    def _insert_add(self, item: int) -> float:
        return self._insert_into(self.add_registers, item)

    def _insert_delete(self, item: int) -> float:
        return self._insert_into(self.del_registers, item)

    def _estimate_one(self, z: float, seen: int) -> float:
        if seen <= 0 or z <= 0.0:
            return 0.0
        estimate = 0.39701 * self.m * (2.0 ** (z / self.m))
        return max(0.0, min(float(max(self.T, self.n_max)), estimate))


class HLLTurnstile(SplitSketchTurnstileMethod):
    method = "HLL-Turnstile"

    def __init__(self, *, T: int, n_max: int, memory_kb: int, epsilon: float, delta: float, seed_hash: int, seed_noise: int):
        super().__init__(
            T=T,
            n_max=n_max,
            memory_kb=memory_kb,
            epsilon=epsilon,
            delta=delta,
            sensitivity=1.0,
            seed_hash=seed_hash,
            seed_noise=seed_noise,
        )
        per_sketch_bits = max(6, self.memory_bits // 2)
        self.m = max(1, per_sketch_bits // 6)
        self.max_rank = 64
        self.add_registers = np.zeros(self.m, dtype=np.uint8)
        self.del_registers = np.zeros(self.m, dtype=np.uint8)
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)
        self.num_cells = 2 * self.m

    def _insert_into(self, registers: np.ndarray, item: int) -> float:
        h1, h2 = hash_pair(item, self.seed_hash)
        row = h1 % self.m
        rank = min(ctz64(h2) + 1, self.max_rank)
        old_reg = int(registers[row])
        if rank > old_reg:
            registers[row] = rank
            return (2.0 ** (-old_reg)) - (2.0 ** (-rank))
        return 0.0

    def _insert_add(self, item: int) -> float:
        return self._insert_into(self.add_registers, item)

    def _insert_delete(self, item: int) -> float:
        return self._insert_into(self.del_registers, item)

    def _estimate_one(self, z: float, seen: int) -> float:
        if seen <= 0:
            return 0.0
        component_cap = float(max(self.T, self.n_max))
        inverse_sum = self.m - z
        if inverse_sum <= 0.0:
            return component_cap
        raw = self.alpha * self.m * self.m / inverse_sum
        return max(0.0, min(component_cap, raw))


class CompressedFlipDPProxy(ScalarGaussianTreeMethod):
    method = "Flip-DP"

    def __init__(self, *, T: int, n_max: int, memory_kb: int, epsilon: float, delta: float, seed_hash: int, seed_noise: int):
        # Memory-matched proxy for the per-element Flip-DP mechanism:
        # keep a deterministic sample of active per-item states and scale it
        # back up.  This mirrors the caveat that the original mechanism is not
        # a kilobyte-memory sketch, while avoiding a pure capacity-bias line
        # that barely responds to epsilon.
        flippancy_bound = 2.0
        H = tree_height(T)
        memory_bits = memory_kb * 1024 * 8
        self.slots = max(16, memory_bits // 64)
        self.sample_rate = min(1.0, max(1.0 / max(1, n_max), 0.8 * self.slots / max(1, n_max)))
        self.sample_threshold = int(self.sample_rate * (MASK64 + 1))
        # Algorithm 3 splits privacy across O(log T) flippancy guesses; fold
        # that composition cost into the memory-matched proxy's tree noise.
        flip_sensitivity = math.sqrt(16.0 * flippancy_bound * (H + 1.0) * (H + 1.0) / H)
        super().__init__(
            T=T,
            n_max=n_max,
            epsilon=epsilon,
            delta=delta,
            sensitivity=flip_sensitivity / self.sample_rate,
            seed_noise=seed_noise,
        )
        self.seed_hash = seed_hash
        self.keys = np.zeros(self.slots, dtype=np.uint64)
        self.active = np.zeros(self.slots, dtype=np.bool_)
        self.flips = np.zeros(self.slots, dtype=np.uint8)
        self.max_probe = 16

    def update(self, t_one_based: int, item: int, op: int) -> None:
        self.current_t = t_one_based
        h1, h2 = hash_pair(item, self.seed_hash)
        if h2 >= self.sample_threshold:
            self.last_z = self.tree.update(t_one_based, 0.0)
            return
        start = h1 % self.slots
        key = np.uint64(h2 or 1)
        delta = 0.0
        chosen = -1
        empty = -1
        for offset in range(self.max_probe):
            slot = (start + offset) % self.slots
            if self.keys[slot] == key:
                chosen = slot
                break
            if self.keys[slot] == 0 and empty < 0:
                empty = slot

        if chosen < 0 and op > 0 and empty >= 0:
            chosen = empty
            self.keys[chosen] = key

        if chosen >= 0:
            if op > 0 and not self.active[chosen] and self.flips[chosen] < 255:
                self.active[chosen] = True
                self.flips[chosen] += 1
                delta = 1.0 / self.sample_rate
            elif op < 0 and self.active[chosen] and self.flips[chosen] < 255:
                self.active[chosen] = False
                self.flips[chosen] += 1
                self.keys[chosen] = 0
                delta = -1.0 / self.sample_rate
        self.last_z = self.tree.update(t_one_based, delta)

    def estimate(self) -> float:
        self.last_estimate = max(0.0, min(self.n_max, self.current_t, self.last_z))
        return self.last_estimate

    def result(self, rmsd: float, final_true: int, elapsed: float) -> MethodResult:
        return self.finish(self.method, rmsd, final_true, elapsed, self.slots)


METHODS = {
    "hfx": HFXGaussian,
    "fm": FMTurnstile,
    "ll": LLTurnstile,
    "hll": HLLTurnstile,
    "flipdp": CompressedFlipDPProxy,
}

METHOD_SEED_OFFSETS = {
    "hfx": 101,
    "fm": 211,
    "ll": 307,
    "hll": 401,
    "flipdp": 503,
}


def synthetic_active_set_events(T: int, n_max: int, p_ins: float, seed: int):
    rng = np.random.default_rng(seed)
    active: list[int] = []
    positions: list[int] = []
    next_id = 1
    active_sum = 0
    max_active = 0

    for _ in range(1, T + 1):
        if not active:
            do_insert = True
        elif len(active) >= n_max:
            do_insert = False
        else:
            do_insert = bool(rng.random() < p_ins)

        if do_insert:
            item = next_id
            next_id += 1
            positions.append(len(active))
            active.append(item)
            op = 1
        else:
            idx = int(rng.integers(0, len(active)))
            item = active[idx]
            last = active[-1]
            active[idx] = last
            positions[last - 1] = idx
            active.pop()
            op = -1

        n_active = len(active)
        active_sum += n_active
        max_active = max(max_active, n_active)
        yield item, op, n_active, max_active, active_sum


def run_one(
    *,
    method_key: str,
    T: int,
    n_max: int,
    p_ins: float,
    memory_kb: int,
    epsilon: float,
    delta: float,
    q: int,
    w: int,
    seed_stream: int,
    seed_hash: int,
    seed_noise: int,
    eval_every: int,
    ignore: int,
    hfx_noise_mode: str,
) -> tuple[MethodResult, int, float]:
    cls = METHODS[method_key]
    kwargs = dict(
        T=T,
        n_max=n_max,
        memory_kb=memory_kb,
        epsilon=epsilon,
        delta=delta,
        seed_hash=seed_hash,
        seed_noise=seed_noise,
    )
    if method_key == "hfx":
        kwargs.update(q=q, w=w, noise_mode=hfx_noise_mode)
    method = cls(**kwargs)

    error_sum_sq = 0.0
    samples = 0
    final_true = 0
    max_active = 0
    active_sum = 0
    start = time.perf_counter()
    for t, (item, op, true_n, max_active, active_sum) in enumerate(
        synthetic_active_set_events(T, n_max, p_ins, seed_stream),
        start=1,
    ):
        method.update(t, item, op)
        final_true = true_n
        if t >= ignore and t % eval_every == 0 and true_n > 0:
            estimate = method.estimate()
            rel = (estimate - true_n) / true_n
            error_sum_sq += rel * rel
            samples += 1
    elapsed = time.perf_counter() - start
    rmsd = math.sqrt(error_sum_sq / samples) if samples else 0.0
    method.estimate()

    if method_key == "hfx":
        result = method.finish(rmsd, final_true, elapsed)
    else:
        result = method.result(rmsd, final_true, elapsed)
    avg_active = active_sum / T if T else 0.0
    return result, max_active, avg_active


def row_key(row: dict[str, object]) -> tuple[int, int, int, int, float, str]:
    q_value = row.get("q", "")
    return (
        int(row["T"]),
        int(row["trial"]),
        -1 if q_value == "" else int(q_value),
        int(row["memory_kb"]),
        float(row["epsilon"]),
        str(row["method"]),
    )


def load_existing_keys(path: Path) -> set[tuple[int, int, int, int, float, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {row_key(row) for row in csv.DictReader(f)}


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_rows(args, output: Optional[Path] = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    completed = load_existing_keys(output) if args.resume and output is not None else set()
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")

    for T in args.T:
        n_max = args.n_max if args.n_max is not None else default_nmax(T)
        eval_every = args.eval_every if args.eval_every is not None else (10_000 if T >= 1_000_000_000 else 1_000)
        q_values = args.q_list if args.q_list is not None else [args.q]
        for trial in range(args.trials):
            seed_stream = args.seed_stream + trial * 10_000
            for q_value in q_values:
                for memory_kb in args.memory_kb:
                    for epsilon in args.epsilon:
                        for method_key in methods:
                            if method_key != "hfx" and args.q_list is not None:
                                continue
                            display_method = METHODS[method_key].method
                            key = (T, trial, q_value if method_key == "hfx" else -1, memory_kb, float(epsilon), display_method)
                            if key in completed:
                                print(
                                    f"Skipping completed {display_method}: "
                                    f"T={T}, mem={memory_kb}KB, eps={epsilon}, q={q_value}, trial={trial + 1}/{args.trials}",
                                    flush=True,
                                )
                                continue
                            seed_hash = args.seed_hash + trial * 100
                            seed_noise = args.seed_noise + trial * 100 + METHOD_SEED_OFFSETS[method_key]
                            print(
                                f"Running {display_method}: "
                                f"T={T}, N_max={n_max}, mem={memory_kb} KB, eps={epsilon}, delta={args.delta}, "
                                f"q={q_value}, R={(1 << q_value) - 1}, rho={epsilon_delta_to_rho(epsilon, args.delta):.8g}, "
                                f"noise_mode={args.hfx_noise_mode}, eval_every={eval_every}, "
                                f"seed_stream={seed_stream}, seed_hash={seed_hash}, seed_noise={seed_noise}, "
                                f"trial={trial + 1}/{args.trials}",
                                flush=True,
                            )
                            result, max_active, avg_active = run_one(
                                method_key=method_key,
                                T=T,
                                n_max=n_max,
                                p_ins=args.p_ins,
                                memory_kb=memory_kb,
                                epsilon=epsilon,
                                delta=args.delta,
                                q=q_value,
                                w=args.w,
                                seed_stream=seed_stream,
                                seed_hash=seed_hash,
                                seed_noise=seed_noise,
                                eval_every=eval_every,
                                ignore=args.ignore,
                                hfx_noise_mode=args.hfx_noise_mode,
                            )
                            rho = epsilon_delta_to_rho(epsilon, args.delta)
                            filtering_rate = result.filtered_updates / T if T else 0.0
                            saturated_ratio = result.saturated_cells / result.num_cells if result.num_cells else 0.0
                            row = {
                                "experiment_name": "experiment1_accuracy_memory",
                                "dataset": "synthetic_active_set",
                                "stream_type": "active_set",
                                "T": T,
                                "num_raw_events": T,
                                "num_set_updates": T,
                                "max_active": max_active,
                                "avg_active": avg_active,
                                "memory_kb": memory_kb,
                                "q": q_value if method_key == "hfx" else "",
                                "R": (1 << q_value) - 1 if method_key == "hfx" else "",
                                "m": result.num_cells // args.w if method_key == "hfx" else result.num_cells,
                                "w": args.w if method_key == "hfx" else "",
                                "num_cells": result.num_cells,
                                "epsilon": epsilon,
                                "delta": args.delta,
                                "rho": rho,
                                "mechanism": "gaussian-" + (args.hfx_noise_mode if method_key == "hfx" else "baseline"),
                                "method": result.method,
                                "seed_hash": seed_hash,
                                "seed_noise": seed_noise,
                                "trial": trial,
                                "rmsd": result.rmsd,
                                "relative_error": result.relative_error,
                                "accepted_updates": result.accepted_updates,
                                "filtered_updates": result.filtered_updates,
                                "filtering_rate": filtering_rate,
                                "saturated_cells": result.saturated_cells,
                                "saturated_cell_ratio": saturated_ratio,
                                "load_p50": "" if result.load_p50 is None else result.load_p50,
                                "load_p95": "" if result.load_p95 is None else result.load_p95,
                                "load_p99": "" if result.load_p99 is None else result.load_p99,
                                "load_max": "" if result.load_max is None else result.load_max,
                                "throughput_mops": result.throughput_mops,
                                "eval_every": eval_every,
                                "p_ins": args.p_ins,
                                "N_max": n_max,
                                "sigma_node": "" if result.sigma_node is None else result.sigma_node,
                                "clipped_evals": result.clipped_evals,
                                "eval_samples": result.eval_samples,
                                "clipping_rate": result.clipped_evals / result.eval_samples if result.eval_samples else 0.0,
                            }
                            if args.resume and output is not None:
                                append_csv_row(output, row)
                                completed.add(row_key(row))
                            else:
                                rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def preset_args(name: str) -> dict[str, object]:
    if name == "smoke":
        return {
            "T": [20_000],
            "memory_kb": [8],
            "epsilon": [1.0],
            "trials": 1,
            "methods": "hfx",
            "eval_every": 500,
            "q": 8,
            "hfx_noise_mode": "load_aware",
        }
    if name == "pilot":
        return {
            "T": [100_000],
            "memory_kb": [8, 16, 32, 64],
            "epsilon": [0.5, 1.0, 2.0, 3.0, 4.0],
            "trials": 1,
            "methods": "hfx,flipdp,fm,ll,hll",
            "eval_every": 1_000,
            "q": 6,
            "hfx_noise_mode": "load_aware",
        }
    if name == "q_sweep":
        return {
            "T": [100_000],
            "memory_kb": [8, 32, 64],
            "epsilon": [1.0],
            "trials": 1,
            "methods": "hfx",
            "eval_every": 1_000,
            "q": 8,
            "hfx_noise_mode": "load_aware",
        }
    if name == "full":
        return {
            "T": [10_000_000, 100_000_000, 1_000_000_000],
            "memory_kb": [8, 16, 32, 64],
            "epsilon": [0.5, 1.0, 2.0, 3.0, 4.0],
            "trials": 10,
            "methods": "hfx,fm,ll,hll,flipdp",
            "eval_every": None,
            "q": 12,
            "hfx_noise_mode": "load_aware",
        }
    raise ValueError(f"Unknown preset: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["smoke", "pilot", "q_sweep", "full"], default="smoke")
    parser.add_argument("--output", default="results/experiment1_active_set.csv")
    parser.add_argument("--T", default=None, help="Comma-separated stream lengths; overrides preset.")
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--p-ins", type=float, default=0.55)
    parser.add_argument("--memory-kb", default=None, help="Comma-separated memory budgets in KB; overrides preset.")
    parser.add_argument("--epsilon", default=None, help="Comma-separated epsilon values; overrides preset.")
    parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--q-list", default=None, help="Comma-separated HFX q values; runs HFX-only q sweep.")
    parser.add_argument("--hfx-noise-mode", choices=["unit_diagnostic", "load_aware", "theorem"], default=None)
    parser.add_argument("--w", type=int, default=32)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--methods", default=None, help="Comma-separated: hfx,fm,ll,hll,flipdp")
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--ignore", type=int, default=1000)
    parser.add_argument("--seed-stream", type=int, default=4801289)
    parser.add_argument("--seed-hash", type=int, default=63242691)
    parser.add_argument("--seed-noise", type=int, default=981723641)
    parser.add_argument("--resume", action="store_true", help="Append each completed setting to --output and skip rows already present.")
    args = parser.parse_args()

    defaults = preset_args(args.preset)
    args.T = parse_number_list(args.T, int) if args.T else defaults["T"]
    args.memory_kb = parse_number_list(args.memory_kb, int) if args.memory_kb else defaults["memory_kb"]
    args.epsilon = parse_number_list(args.epsilon, float) if args.epsilon else defaults["epsilon"]
    args.trials = args.trials if args.trials is not None else defaults["trials"]
    args.methods = args.methods if args.methods is not None else defaults["methods"]
    args.q = args.q if args.q is not None else defaults["q"]
    args.hfx_noise_mode = args.hfx_noise_mode if args.hfx_noise_mode is not None else defaults["hfx_noise_mode"]
    args.q_list = parse_number_list(args.q_list, int) if args.q_list else None
    if args.preset == "q_sweep" and args.q_list is None:
        args.q_list = [4, 6, 8, 12, 16]
    if args.eval_every is None:
        args.eval_every = defaults["eval_every"]
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    rows = build_rows(args, output=output)
    if not args.resume:
        write_csv(output, rows)
        print(f"Wrote {len(rows)} rows to {output.resolve()}")
    else:
        print(f"Resume run finished; results are in {output.resolve()}")


if __name__ == "__main__":
    main()
